//! Derive the launcher `ParamDef` schema directly from the config types, so a
//! field and its schema entry cannot drift (new-interface-design §11).
//!
//! Two derives, both emitting an inherent `const` that `schema::dump::list_params`
//! aggregates:
//!   - `#[derive(ParamStruct)]` on a plain struct → `pub const PARAMS: &[ParamDef]`
//!     (one entry per non-skipped field). Used for the `*_COMMON` blocks
//!     (`ModelSpec`, `GroupSpec`, `PoolSpec`, `WorkloadSpec`, `IoSpec`).
//!   - `#[derive(ProviderSchema)]` on a `#[serde(tag = "type")]` enum →
//!     `pub const SCHEMA: &[(&str, &[ParamDef])]`, one row per variant: the
//!     serde-snake_case tag plus that variant's own (non-flatten) field params.
//!
//! Name + wire type are inferred from the field's Rust type; the description is
//! the field's `///` doc comment. Everything serde/clap cannot express rides on
//! an inert `#[param(...)]` helper:
//!   - `#[param(skip)]`             → field contributes no param (nested
//!     sub-trees: `groups`, `arch`, `worker`).
//!   - `#[param(default = LIT)]`    → `.default_<kind>(LIT)`.
//!   - `#[param(cache_key)]`        → `.cache_key()`.
//!   - `#[param(choices = CONST)]`  → `.choices(&CONST)`.
//!   - `#[param(string)]`           → treat the field as a `string` param even
//!     though its Rust type is a foreign enum
//!     (`placement`, `log_level`, `batch_policy`).
//!
//! `#[serde(flatten)]` fields are skipped automatically (the flattened struct
//! contributes its own `PARAMS` block). A `bool` with no explicit default gets an
//! implicit `false` (absent flag = off); an `Option<T>` with no default is marked
//! `.optional()`. A `Vec<T>` is a list param that is required UNLESS the field
//! also carries `#[serde(default)]` — in that case (and for any other type
//! tagged `#[serde(default)]`) the param is marked `.optional()`, matching
//! serde's "absent = Default::default()" semantics (e.g. `Vec<f32>` defaults to
//! the empty list).

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, punctuated::Punctuated, spanned::Spanned, Attribute, Data, DeriveInput,
    Expr, ExprLit, Field, Fields, GenericArgument, Lit, Meta, PathArguments, Token, Type,
};

#[derive(Clone, Copy)]
enum Scalar {
    Int,
    Float,
    Bool,
    Str,
    Path,
}

/// What a field maps to on the `ParamDef` side.
struct Classified {
    ctor: &'static str,
    /// `default_*` setter name, or "" for list params (no scalar default).
    default_method: &'static str,
    optional: bool,
    is_bool: bool,
}

/// Parsed `#[param(...)]` helper attribute.
#[derive(Default)]
struct ParamAttr {
    skip: bool,
    force_string: bool,
    cache_key: bool,
    default: Option<Lit>,
    choices: Option<Expr>,
}

// ── struct derive → `PARAMS` ────────────────────────────────────────────────

#[proc_macro_derive(ParamStruct, attributes(param))]
pub fn derive_param_struct(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            _ => return err(name, "ParamStruct needs a struct with named fields"),
        },
        _ => return err(name, "ParamStruct can only be derived on structs"),
    };

    let mut errors = Vec::new();
    let defs = param_defs(fields.iter(), &mut errors);
    if let Some(e) = combine(errors) {
        return e.to_compile_error().into();
    }

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            pub const PARAMS: &'static [::simulator::schema::ParamDef] = &[ #( #defs ),* ];
        }
    }
    .into()
}

// ── enum derive → `SCHEMA` ──────────────────────────────────────────────────

#[proc_macro_derive(ProviderSchema, attributes(param))]
pub fn derive_provider_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let variants = match &input.data {
        Data::Enum(e) => &e.variants,
        _ => return err(name, "ProviderSchema can only be derived on enums"),
    };

    let mut errors = Vec::new();
    let mut rows = Vec::new();
    for v in variants {
        let tag = snake_case(&v.ident.to_string());
        let fields = match &v.fields {
            Fields::Named(named) => named.named.iter().collect::<Vec<_>>(),
            Fields::Unit => Vec::new(),
            Fields::Unnamed(_) => {
                errors.push(syn::Error::new(
                    v.span(),
                    "ProviderSchema needs struct-like or unit variants",
                ));
                continue;
            }
        };
        let defs = param_defs(fields.into_iter(), &mut errors);
        rows.push(quote! { ( #tag, &[ #( #defs ),* ] ) });
    }
    if let Some(e) = combine(errors) {
        return e.to_compile_error().into();
    }

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            pub const SCHEMA: &'static [(&'static str, &'static [::simulator::schema::ParamDef])] =
                &[ #( #rows ),* ];
        }
    }
    .into()
}

// ── shared: one field → a `ParamDef` builder chain ──────────────────────────

fn param_defs<'a>(
    fields: impl Iterator<Item = &'a Field>,
    errors: &mut Vec<syn::Error>,
) -> Vec<proc_macro2::TokenStream> {
    let mut defs = Vec::new();
    for f in fields {
        if is_serde_flatten(&f.attrs) {
            continue;
        }
        let attr = match parse_param_attr(&f.attrs) {
            Ok(a) => a,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };
        if attr.skip {
            continue;
        }
        let field = f.ident.as_ref().expect("named fields enforced by callers");
        let name_str = field.to_string();

        // `#[param(string)]` overrides type inference (the Rust field is a
        // foreign enum but the schema treats it as a `string` with `choices`).
        let mut cls = if attr.force_string {
            Classified {
                ctor: "string",
                default_method: "default_string",
                optional: false,
                is_bool: false,
            }
        } else {
            match classify(&f.ty) {
                Ok(c) => c,
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            }
        };
        // `#[serde(default)]` declares "absent = Default::default()" at the
        // deserializer level; surface that as an optional schema param so the
        // launcher validator doesn't insist the user spell it out. Lets a
        // `Vec<T>` with a sensible empty default (e.g. routing_profile) stay
        // omittable in YAML.
        if is_serde_default(&f.attrs) {
            cls.optional = true;
        }

        let ctor = syn::Ident::new(cls.ctor, f.span());
        let mut chain = quote! { ::simulator::schema::ParamDef::#ctor(#name_str) };

        if let Some(lit) = attr.default {
            if cls.default_method.is_empty() {
                errors.push(syn::Error::new(
                    f.ty.span(),
                    "param: a list param cannot carry a scalar default",
                ));
                continue;
            }
            let setter = syn::Ident::new(cls.default_method, f.span());
            chain = quote! { #chain.#setter(#lit) };
        } else if cls.is_bool {
            // A bare `bool` flag is off when absent; serde carries no default, so
            // inject the implicit `false` (else it would serialize as required).
            chain = quote! { #chain.default_bool(false) };
        } else if cls.optional {
            chain = quote! { #chain.optional() };
        }
        if let Some(choices) = attr.choices {
            chain = quote! { #chain.choices(&#choices) };
        }
        if attr.cache_key {
            chain = quote! { #chain.cache_key() };
        }
        let desc = doc_string(&f.attrs);
        chain = quote! { #chain.desc(#desc) };
        defs.push(chain);
    }
    defs
}

/// Map a field type to its `ParamDef` shape, unwrapping `Option<T>` / `Vec<T>`.
fn classify(ty: &Type) -> syn::Result<Classified> {
    if let Some(inner) = generic_inner(ty, "Option") {
        let mut c = classify_scalar(inner)?;
        c.optional = true;
        return Ok(c);
    }
    if let Some(inner) = generic_inner(ty, "Vec") {
        return Ok(Classified {
            ctor: list_ctor(scalar_kind(inner)?, inner)?,
            default_method: "",
            optional: false,
            is_bool: false,
        });
    }
    classify_scalar(ty)
}

fn classify_scalar(ty: &Type) -> syn::Result<Classified> {
    let s = scalar_kind(ty)?;
    Ok(Classified {
        ctor: scalar_ctor(s),
        default_method: scalar_default(s),
        optional: false,
        is_bool: matches!(s, Scalar::Bool),
    })
}

fn scalar_kind(ty: &Type) -> syn::Result<Scalar> {
    let ident = last_ident(ty)
        .ok_or_else(|| syn::Error::new(ty.span(), "param: unsupported field type"))?;
    match ident.as_str() {
        "f32" | "f64" => Ok(Scalar::Float),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128"
        | "isize" => Ok(Scalar::Int),
        "bool" => Ok(Scalar::Bool),
        "String" => Ok(Scalar::Str),
        "PathBuf" => Ok(Scalar::Path),
        other => Err(syn::Error::new(
            ty.span(),
            format!("param: unsupported field type `{other}` (add `#[param(string)]` or `#[param(skip)]`)"),
        )),
    }
}

fn scalar_ctor(s: Scalar) -> &'static str {
    match s {
        Scalar::Int => "int",
        Scalar::Float => "float",
        Scalar::Bool => "bool",
        Scalar::Str => "string",
        Scalar::Path => "path",
    }
}

fn scalar_default(s: Scalar) -> &'static str {
    match s {
        Scalar::Int => "default_int",
        Scalar::Float => "default_float",
        Scalar::Bool => "default_bool",
        Scalar::Str => "default_string",
        Scalar::Path => "default_path",
    }
}

fn list_ctor(s: Scalar, ty: &Type) -> syn::Result<&'static str> {
    match s {
        Scalar::Int => Ok("int_list"),
        Scalar::Float => Ok("float_list"),
        Scalar::Str => Ok("string_list"),
        Scalar::Path => Ok("path_list"),
        Scalar::Bool => Err(syn::Error::new(
            ty.span(),
            "param: no bool_list ParamType (Vec<bool> unsupported)",
        )),
    }
}

fn last_ident(ty: &Type) -> Option<String> {
    if let Type::Path(tp) = ty {
        return tp.path.segments.last().map(|s| s.ident.to_string());
    }
    None
}

fn generic_inner<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
    let Type::Path(tp) = ty else { return None };
    let seg = tp.path.segments.last()?;
    if seg.ident != wrapper {
        return None;
    }
    let PathArguments::AngleBracketed(ab) = &seg.arguments else {
        return None;
    };
    match ab.args.first()? {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

/// serde `RenameRule::SnakeCase` for a PascalCase variant ident — insert `_`
/// before each non-leading uppercase, lowercase everything (matches the wire tag
/// produced by `#[serde(rename_all = "snake_case")]`).
fn snake_case(ident: &str) -> String {
    let mut out = String::new();
    for (i, ch) in ident.char_indices() {
        if i > 0 && ch.is_uppercase() {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

/// Join `///` doc lines (each trimmed) into the `ParamDef` description.
fn doc_string(attrs: &[Attribute]) -> String {
    let mut parts = Vec::new();
    for a in attrs {
        if !a.path().is_ident("doc") {
            continue;
        }
        if let Meta::NameValue(nv) = &a.meta {
            if let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &nv.value
            {
                parts.push(s.value().trim().to_string());
            }
        }
    }
    parts.join(" ")
}

/// Parse the inert `#[param(...)]` helper attribute(s) on a field.
fn parse_param_attr(attrs: &[Attribute]) -> syn::Result<ParamAttr> {
    let mut out = ParamAttr::default();
    for a in attrs {
        if !a.path().is_ident("param") {
            continue;
        }
        let nested = a.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;
        for m in nested {
            match m {
                Meta::Path(p) if p.is_ident("skip") => out.skip = true,
                Meta::Path(p) if p.is_ident("string") => out.force_string = true,
                Meta::Path(p) if p.is_ident("cache_key") => out.cache_key = true,
                Meta::NameValue(nv) if nv.path.is_ident("default") => {
                    if let Expr::Lit(ExprLit { lit, .. }) = nv.value {
                        out.default = Some(lit);
                    } else {
                        return Err(syn::Error::new(
                            nv.value.span(),
                            "param: default must be a literal",
                        ));
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("choices") => out.choices = Some(nv.value),
                other => return Err(syn::Error::new(other.span(), "param: unknown key")),
            }
        }
    }
    Ok(out)
}

fn is_serde_flatten(attrs: &[Attribute]) -> bool {
    serde_has_marker(attrs, "flatten")
}

/// `true` iff the field carries `#[serde(default)]` (bare path, no `=`). We
/// don't try to interpret `#[serde(default = "fn")]` here — that already means
/// "deserializer fills it", which is the same optional intent at the schema
/// layer.
fn is_serde_default(attrs: &[Attribute]) -> bool {
    serde_has_marker(attrs, "default")
}

fn serde_has_marker(attrs: &[Attribute], marker: &str) -> bool {
    for a in attrs {
        if !a.path().is_ident("serde") {
            continue;
        }
        if let Ok(nested) = a.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
            if nested.iter().any(|m| m.path().is_ident(marker)) {
                return true;
            }
        }
    }
    false
}

fn combine(errors: Vec<syn::Error>) -> Option<syn::Error> {
    let mut it = errors.into_iter();
    let mut first = it.next()?;
    for e in it {
        first.combine(e);
    }
    Some(first)
}

fn err(tokens: &impl ToTokens, msg: &str) -> TokenStream {
    syn::Error::new_spanned(tokens, msg)
        .to_compile_error()
        .into()
}
