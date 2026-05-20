//! Proc-macro support for L1 kernel Config / Input structs.
//!
//! `#[derive(SweepCoords)]` generates the mechanical "flatten Input fields to
//! `Vec<f64>`" conversion that every kernel needs at lookup time. Field order
//! follows declaration order, each field is cast `as f64`. Non-castable field
//! types (String, bool, struct, etc.) produce a clear "no `as` conversion"
//! error at the derive site, telling the kernel author to put that field in
//! `Config` instead.
//!
//! `#[derive(KernelConfig)]` generates the `KernelConfig` trait impl that
//! exposes the required `backends: Vec<&'static str>` field. Structs that
//! don't have a `backends` field fail at the generated `&self.backends`
//! access; the proc-macro itself just plugs the field name through.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(SweepCoords)]
pub fn derive_sweep_coords(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            Fields::Unnamed(_) | Fields::Unit => {
                return syn::Error::new_spanned(
                    &input.ident,
                    "SweepCoords can only be derived on structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        Data::Enum(_) | Data::Union(_) => {
            return syn::Error::new_spanned(
                &input.ident,
                "SweepCoords can only be derived on structs",
            )
            .to_compile_error()
            .into();
        }
    };

    let coord_exprs = fields.iter().map(|f| {
        let field_name = f.ident.as_ref().expect("named fields enforced above");
        quote! { (self.#field_name) as f64 }
    });

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::simulator::timing::SweepCoords for #name #ty_generics #where_clause {
            fn coords(&self) -> ::simulator::timing::Coords {
                ::simulator::timing::Coords::new([ #( #coord_exprs ),* ])
            }
        }
    };

    expanded.into()
}

#[proc_macro_derive(KernelConfig)]
pub fn derive_kernel_config(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let backends_field_label = format!("{name}.backends");

    if !matches!(
        &input.data,
        Data::Struct(s) if matches!(s.fields, Fields::Named(_))
    ) {
        return syn::Error::new_spanned(
            name,
            "KernelConfig can only be derived on structs with named fields",
        )
        .to_compile_error()
        .into();
    }

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::simulator::timing::KernelConfig for #name #ty_generics #where_clause {
            fn backends(&self) -> &[&'static str] {
                &self.backends
            }
            const BACKENDS_FIELD: &'static str = #backends_field_label;
        }
    };

    expanded.into()
}
