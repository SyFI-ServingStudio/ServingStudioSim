//! `#[derive(DeploymentParams)]` — single-source bridge from a clap `Args`
//! struct to its `ParamDef` schema slice (design §1.8.3/§1.8.4).
//!
//! The clap struct stays an ordinary `#[derive(clap::Args)]` with normal
//! `#[arg(...)]` fields. This derive reads those same fields and emits two
//! consts so the CLI surface and the `list-params` schema cannot drift:
//!   - inherent `OWN_PARAMS: &[ParamDef]` — this struct's own (non-flatten)
//!     params, used when the struct is flattened into another;
//!   - `impl ParamSchema { PARAM_GROUPS: &[&[ParamDef]] }` — the full schema,
//!     composing each `#[command(flatten)]` fragment's `OWN_PARAMS` (in field
//!     order) followed by this struct's own group. A `Deployment` defaults its
//!     `PARAM_GROUPS` to `Args::PARAM_GROUPS`, so it never re-lists fragments.
//!
//! Name / type / default / required are all inferred from the field's Rust type
//! and its existing `#[arg(...)]`, so they are written exactly once. Metadata
//! clap has no notion of rides on an inert `#[param(...)]` helper attribute:
//!   - `#[param(cache_key)]`       → `ParamDef::cache_key()`
//!   - `#[param(choices = CONST)]` → `ParamDef::choices(&CONST)`
//!
//! Adding a new field type is one arm in `scalar_kind` — additive, not the
//! combinatorial arm growth a `macro_rules!` muncher would hit.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, punctuated::Punctuated, spanned::Spanned, Attribute, Data, DeriveInput,
    Expr, ExprLit, Fields, GenericArgument, Lit, Meta, PathArguments, Token, Type,
};

#[derive(Clone, Copy)]
enum Scalar {
    Int,
    Float,
    Bool,
    Str,
    Path,
}

/// What a field maps to on the `ParamDef` side: which constructor, which
/// `default_*` setter, whether it is omittable (`Option<T>` / `Vec<T>`), and
/// whether it is a `bool` (clap flag → implicit `false` default).
struct Classified {
    ctor: &'static str,
    default_method: &'static str,
    optional: bool,
    is_bool: bool,
}

#[proc_macro_derive(DeploymentParams, attributes(param))]
pub fn derive_deployment_params(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            Fields::Unnamed(_) | Fields::Unit => {
                return err(name, "DeploymentParams needs a struct with named fields");
            }
        },
        Data::Enum(_) | Data::Union(_) => {
            return err(name, "DeploymentParams can only be derived on structs");
        }
    };

    let mut defs = Vec::new();
    let mut flatten_tys: Vec<&Type> = Vec::new();
    let mut errors: Vec<syn::Error> = Vec::new();

    for f in fields {
        if is_flatten(&f.attrs) {
            flatten_tys.push(&f.ty);
            continue;
        }
        let field = f.ident.as_ref().expect("named fields enforced above");
        let name_str = field.to_string();

        let cls = match classify(&f.ty) {
            Ok(c) => c,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };
        let desc = doc_string(&f.attrs);
        let default = arg_default(&f.attrs);
        let (cache_key, choices) = param_meta(&f.attrs);

        let ctor = format_ident!("{}", cls.ctor);
        let mut chain = quote! { ::simulator::schema::ParamDef::#ctor(#name_str) };

        if let Some(lit) = default {
            if cls.default_method.is_empty() {
                errors.push(syn::Error::new(
                    f.ty.span(),
                    "DeploymentParams: list params cannot carry a scalar default",
                ));
                continue;
            }
            let setter = format_ident!("{}", cls.default_method);
            chain = quote! { #chain.#setter(#lit) };
        } else if cls.is_bool && !cls.optional {
            // A clap `bool` is a flag: present => true, absent => false. clap
            // carries no `default_value_t`, so inject the implicit false here
            // (otherwise a defaultless bool would wrongly serialize as required).
            chain = quote! { #chain.default_bool(false) };
        } else if cls.optional {
            chain = quote! { #chain.optional() };
        }
        if let Some(choices) = choices {
            chain = quote! { #chain.choices(&#choices) };
        }
        if cache_key {
            chain = quote! { #chain.cache_key() };
        }
        chain = quote! { #chain.desc(#desc) };
        defs.push(chain);
    }

    if let Some(combined) = combine(errors) {
        return combined.to_compile_error().into();
    }

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Flattened fragments contribute their `OWN_PARAMS`; this struct's own
    // params follow as the final group (mirrors clap field order). Composing
    // here lets a `Deployment` default `PARAM_GROUPS` to `Args::PARAM_GROUPS`
    // rather than re-listing the fragments by hand.
    let flatten_groups = flatten_tys.iter().map(|ty| quote! { #ty::OWN_PARAMS, });
    let own_group = if defs.is_empty() {
        quote! {}
    } else {
        quote! { Self::OWN_PARAMS, }
    };

    quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            pub const OWN_PARAMS: &'static [::simulator::schema::ParamDef] = &[ #( #defs ),* ];
        }

        impl #impl_generics ::simulator::schema::ParamSchema for #name #ty_generics #where_clause {
            const PARAM_GROUPS: &'static [&'static [::simulator::schema::ParamDef]] =
                &[ #( #flatten_groups )* #own_group ];
        }
    }
    .into()
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
            optional: true,
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
    let ident = last_ident(ty).ok_or_else(|| {
        syn::Error::new(ty.span(), "DeploymentParams: unsupported field type")
    })?;
    match ident.as_str() {
        "f32" | "f64" => Ok(Scalar::Float),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128"
        | "isize" => Ok(Scalar::Int),
        "bool" => Ok(Scalar::Bool),
        "String" => Ok(Scalar::Str),
        "PathBuf" => Ok(Scalar::Path),
        other => Err(syn::Error::new(
            ty.span(),
            format!("DeploymentParams: unsupported field type `{other}`"),
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
            "DeploymentParams: no bool_list ParamType (Vec<bool> unsupported)",
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

/// Join `///` doc lines (each trimmed) into the `ParamDef` description, matching
/// how clap derives `--help` from the same comments.
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

/// Pull a literal default out of the existing `#[arg(... default_value[_t] = LIT)]`
/// so the default is declared once (on the clap side) and reused here.
fn arg_default(attrs: &[Attribute]) -> Option<Lit> {
    for a in attrs {
        if !a.path().is_ident("arg") {
            continue;
        }
        let Ok(nested) = a.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) else {
            continue;
        };
        for m in nested {
            if let Meta::NameValue(nv) = m {
                if nv.path.is_ident("default_value_t") || nv.path.is_ident("default_value") {
                    if let Expr::Lit(ExprLit { lit, .. }) = nv.value {
                        return Some(lit);
                    }
                }
            }
        }
    }
    None
}

/// Read the inert `#[param(...)]` helper: `(cache_key, choices_const_expr)`.
fn param_meta(attrs: &[Attribute]) -> (bool, Option<Expr>) {
    let mut cache_key = false;
    let mut choices = None;
    for a in attrs {
        if !a.path().is_ident("param") {
            continue;
        }
        let Ok(nested) = a.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) else {
            continue;
        };
        for m in nested {
            match m {
                Meta::Path(p) if p.is_ident("cache_key") => cache_key = true,
                Meta::NameValue(nv) if nv.path.is_ident("choices") => choices = Some(nv.value),
                _ => {}
            }
        }
    }
    (cache_key, choices)
}

fn is_flatten(attrs: &[Attribute]) -> bool {
    for a in attrs {
        if !(a.path().is_ident("command") || a.path().is_ident("clap")) {
            continue;
        }
        if let Ok(nested) = a.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
            if nested.iter().any(|m| m.path().is_ident("flatten")) {
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

fn err(tokens: &impl quote::ToTokens, msg: &str) -> TokenStream {
    syn::Error::new_spanned(tokens, msg).to_compile_error().into()
}
