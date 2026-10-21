// Copyright (c) 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::*;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ident;
use crate::isa::*;

fn bitmask(n: u8) -> u64 {
    (1 << n) - 1
}

#[derive(Clone, Copy)]
struct DecoderVal<'a> {
    fixed_mask: u64,
    fixed_value: u64,
    instr: &'a Instr,
}

impl DecoderVal<'_> {
    fn new(instr: &Instr) -> DecoderVal<'_> {
        let (fixed_mask, fixed_value) = instr
            .fields
            .iter()
            .filter_map(|f| match f {
                InstrField::Physical(PhysicalField {
                    exact: Some(v),
                    bits,
                    ..
                }) => Some((
                    bitmask(bits.end - bits.start) << bits.start,
                    v << bits.start,
                )),
                _ => None,
            })
            .fold((0, 0), |(amask, aval), (mask, value)| {
                (amask | mask, aval | value)
            });

        DecoderVal {
            instr,
            fixed_mask,
            fixed_value,
        }
    }

    fn get_specificity(&self) -> u32 {
        self.fixed_mask.count_ones()
    }
}

impl Ord for DecoderVal<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_specificity()
            .cmp(&other.get_specificity())
            .reverse()
            .then(self.instr.name.as_str().cmp(other.instr.name.as_str()))
    }
}

impl PartialOrd for DecoderVal<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DecoderVal<'_> {
    fn eq(&self, other: &Self) -> bool {
        self <= other && self >= other
    }
}

impl Eq for DecoderVal<'_> {}

enum DecoderNode<'a> {
    Leaf {
        options: Vec<DecoderVal<'a>>,
    },
    Internal {
        mask: u64,
        children: HashMap<u64, Box<DecoderNode<'a>>>,
    },
}

impl DecoderNode<'_> {
    fn build<'a>(values: Vec<&'a Instr>) -> DecoderNode<'a> {
        return DecoderNode::build_internal(
            values.into_iter().map(DecoderVal::new).collect(),
            0,
        );
    }

    fn build_internal<'a>(
        values: Vec<DecoderVal<'a>>,
        already_used_bits: u64,
    ) -> DecoderNode<'a> {
        let mut biggest_mask = !(0 as u64);
        for v in &values {
            biggest_mask &= v.fixed_mask;
        }
        biggest_mask &= !already_used_bits;

        let mut buckets: HashMap<u64, Vec<DecoderVal>> = Default::default();
        for v in values {
            let key = v.fixed_value & biggest_mask;
            buckets.entry(key).or_insert(vec![]).push(v);
        }

        if buckets.len() == 1 {
            let mut all_values: Vec<DecoderVal> =
                buckets.into_values().next().unwrap();
            all_values.sort();

            return DecoderNode::Leaf {
                options: all_values,
            };
        }

        let mut children: HashMap<u64, Box<DecoderNode>> = Default::default();
        for (key, sub_values) in buckets {
            let node = Box::new(DecoderNode::build_internal(
                sub_values,
                already_used_bits | biggest_mask,
            ));
            children.insert(key, node);
        }

        DecoderNode::Internal {
            mask: biggest_mask,
            children: children,
        }
    }

    fn to_tokens<F, G>(&self, elem_cb: &F, check_cb: &G) -> TokenStream
    where
        F: Fn(&Instr) -> TokenStream,
        G: Fn(&Instr) -> TokenStream,
    {
        match self {
            DecoderNode::Leaf { options } => {
                let cases: TokenStream = options
                    .into_iter()
                    .map(|option| {
                        let fixed_mask = option.fixed_mask;
                        let expected_masked =
                            option.fixed_value & option.fixed_mask;
                        let check_ts = check_cb(option.instr);
                        let val_ts = elem_cb(option.instr);
                        quote! {
                            if (input_value & #fixed_mask) == #expected_masked {
                                result = #check_ts.map(|_| #val_ts);
                                if result.is_ok() {
                                    return result;
                                }
                            }
                        }
                    })
                    .collect();

                quote! {
                    {
                        let mut result = Err(InvalidInstrError::Any);
                        #cases
                        return result;
                    }
                }
            }
            DecoderNode::Internal { mask, children } => {
                // For better readability, sort the keys for the output.
                let mut sorted: Vec<_> = children.keys().collect();
                sorted.sort();

                let cases: TokenStream = sorted
                    .into_iter()
                    .map(|key| {
                        let child_impl = children
                            .get(key)
                            .expect("Key must exist")
                            .to_tokens(elem_cb, check_cb);
                        quote! {
                            #key => #child_impl,
                        }
                    })
                    .collect();

                quote! {
                    match input_value & #mask {
                        #cases
                        _ => Err(InvalidInstrError::Any)
                    }
                }
            }
        }
    }
}

struct LoadField<'a> {
    name: String,
    field: &'a InstrField,
}

impl LoadField<'_> {
    const DEFAULT_LOAD_TYPE: FieldType = FieldType::Uint(32);

    fn new<'a>(name: &str, instr: &'a Instr) -> LoadField<'a> {
        let field = instr.get_named_field(name).unwrap();

        LoadField {
            name: String::from(name),
            field,
        }
    }

    fn print_from_alias(&self) -> bool {
        matches!(self.field.field_type(), Some(FieldType::Enum(_)))
    }

    fn ident_print(&self) -> Ident {
        if self.print_from_alias() {
            Ident::new(&format!("{}_", &self.name), Span::call_site())
        } else {
            Ident::new(&self.name, Span::call_site())
        }
    }

    fn as_field_type_ts(&self, id_in: &Ident) -> TokenStream {
        let field_type =
            self.field.field_type().unwrap_or(&Self::DEFAULT_LOAD_TYPE);

        match field_type {
            FieldType::Enum(e) => {
                let ei = &e.ident;
                // All enums implement TryDecode<u8>
                quote! { #ei::try_decode((#id_in) as u8, arch)? }
            }
            FieldType::Int(n) | FieldType::PcRelOffsetSigned(n) => {
                // n == "physical size" + mod extra bits
                // The mod has been reversed already, thus the "encoded" size
                // here is exactly n.
                quote! {sign_ext(#id_in, #n)}
            }
            FieldType::Source
            | FieldType::Source64
            | FieldType::Uint(_)
            | FieldType::PcRelOffsetUnsigned(_) => quote! {#id_in},
        }
    }

    fn is_valid_to_tokens(&self, id_in: &Ident, ts: &mut TokenStream) {
        let field_type =
            self.field.field_type().unwrap_or(&Self::DEFAULT_LOAD_TYPE);

        ts.extend(match field_type {
            FieldType::Enum(e) => {
                if let Some(restrict) = self.field.restrict() {
                    let ei = &e.ident;
                    let mut ts: TokenStream = Default::default();

                    for allowed_value in &restrict.values {
                        assert!(
                            format!("{}", allowed_value.enum_type.ident())
                                == format!("{}", &e.ident),
                        );
                        let ev = &allowed_value.value_ident;
                        ts.extend(quote! {#ei::#ev,});
                    }

                    quote! {if ! [#ts].contains(&#id_in) {
                        return Err(InvalidInstrError::Any.into());
                    }}
                } else {
                    quote! {}
                }
            }
            FieldType::Int(_)
            | FieldType::Uint(_)
            | FieldType::PcRelOffsetSigned(_)
            | FieldType::PcRelOffsetUnsigned(_) => quote! {}, // TODO
            FieldType::Source | FieldType::Source64 => quote! {}, // TODO
        });
    }
}

impl ToTokens for LoadField<'_> {
    fn to_tokens(&self, ts: &mut TokenStream) {
        let raw_load_ts = match self.field {
            InstrField::Physical(f) => {
                let mask = bitmask(f.bits.end - f.bits.start);
                let start = f.bits.start;
                let mut load_ts = quote! {((v >> #start) & #mask)};

                load_ts = match &f.mod_ {
                    FieldMod::Shr(n) => quote! {(#load_ts << #n)},
                    FieldMod::Minus(n) => quote! {(#load_ts + #n)},
                    FieldMod::Align(_) | FieldMod::None => load_ts,
                };

                // All values we unpack will fit into u32.
                assert!(f.bits.len() + usize::from(f.mod_.extra_bits()) <= 32);
                quote! { (#load_ts as u32) }
            }
            InstrField::Virtual(f) => f.expr.to_token_stream(),
            InstrField::Reserved(_) => unreachable!("Can't load reserved"),
        };

        let ident = Ident::new(self.field.name().unwrap(), Span::call_site());
        ts.extend(quote! {let #ident = #raw_load_ts; });

        let cast_ts = self.as_field_type_ts(&ident);
        let cast_ident = self.ident_print();
        ts.extend(quote! {let #cast_ident = #cast_ts;});

        self.is_valid_to_tokens(&cast_ident, ts);
    }
}

struct PrintAsStaging {
    count_id: Ident,
    offset: u8,
}

impl PrintAsStaging {
    fn new(input: bool, instr: &Instr) -> PrintAsStaging {
        let count_field = instr
            .get_sr_count_field(input)
            .expect("Missing sr count field");

        let count_id = count_field
            .ident()
            .cloned()
            .expect("Missing ident on sr count field");

        // TODO: Resolve fields with "exact" values to their constant value.
        // Make PrintAsStaging an enum and have a "Const" variant.

        // sr_count_t starts from 0. sr_write_count_t and ls_multi_sr_count start
        // from 1, meaning u8 == 0, means 1 reg is written.
        let offset: u8 = match count_field.field_type() {
            Some(FieldType::Enum(e))
                if e.name == "sr_write_count_t"
                    || e.name == "ls_multi_sr_count_m" =>
            {
                1
            }
            _ => 0,
        };

        PrintAsStaging { count_id, offset }
    }
}

pub enum ImmType {
    Uint,
    Int,
    Float,
    V2F16,
    V2I16,
    V4I8,
}

impl From<instr::ImmType> for ImmType {
    fn from(t: instr::ImmType) -> ImmType {
        match t {
            instr::ImmType::Uint => ImmType::Uint,
            instr::ImmType::Int => ImmType::Int,
            instr::ImmType::Float => ImmType::Float,
            instr::ImmType::V2F16 => ImmType::V2F16,
            instr::ImmType::V2I16 => ImmType::V2I16,
            instr::ImmType::V4I8 => ImmType::V4I8,
        }
    }
}

enum PrintAs {
    Id,
    Imm(ImmType),
    Src(bool),
    Dst(bool),
    Staging(PrintAsStaging),
    PcRelOffset,
}

impl PrintAs {
    fn to_tokens(&self, ident: &Ident, ts: &mut TokenStream) {
        match &self {
            PrintAs::Id => (),
            PrintAs::Dst(is64) => {
                ts.extend(quote! {
                    let #ident = PrintDst { reg: (#ident as u8), is64: #is64 };
                });
            }
            PrintAs::Imm(t) => {
                ts.extend(match t {
                    ImmType::V4I8 | ImmType::V2F16 | ImmType::V2I16 | // TODO Treat these correctly
                    ImmType::Float | ImmType::Uint | ImmType::Int => {
                        quote! { let #ident = PrintImm{ val: #ident }; }
                    }
                });
            }
            PrintAs::Src(is64) => {
                let n = if *is64 {
                    "SourceEncoding64"
                } else {
                    "SourceEncoding"
                };
                let cls_id = Ident::new(n, Span::call_site());
                ts.extend(quote! { let #ident = #cls_id::try_decode(
                #ident as u8, arch, (fau_page_index as u8), ctx.fau32)?; });
            }
            PrintAs::Staging(s) => {
                let count_id = &s.count_id;
                let offset = s.offset;
                ts.extend(quote! {
                    assert!(#ident <= u8::MAX as u32);
                    assert!(#count_id <= u8::MAX as u32);
                    let #ident = PrintStaging {
                        regs: (#ident as u8)..(
                                  (#ident as u8) + (#count_id as u8) + #offset)
                    };
                });
            }
            PrintAs::PcRelOffset => {
                ts.extend(quote!{
                    let #ident = PrintPcRelOffset{ val: #ident, pc_base: ctx.pc_base };
                });
            }
        };
    }

    fn declare_types(ts: &mut TokenStream) {
        ts.extend(quote! {
            struct PrintDst {
                reg: u8,
                is64: bool,
            }
            impl std::fmt::Display for PrintDst {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    if self.is64 {
                        write!(f, "[r{}:r{}]", self.reg, self.reg + 1)
                    } else {
                        write!(f, "r{}", self.reg)
                    }
               }
            }

            struct PrintImm<T> {
                val: T,
            }

            impl std::fmt::Display for PrintImm<i32>
            {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    if self.val < 0 {
                        write!(f, "#-{:#x}", -self.val)
                    } else {
                        write!(f, "#{:#x}", self.val)
                    }
                }
            }

            impl std::fmt::Display for PrintImm<u32>
            {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "#{:#x}", self.val)
                }
            }

            struct PrintStaging {
                regs: core::ops::Range<u8>,
            }

            impl std::fmt::Display for PrintStaging {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "@r{}", self.regs.start)?;
                    for r in ((self.regs.start + 1)..(self.regs.end)) {
                        write!(f, ":r{}", r)?;
                    }
                    Ok(())
                }
            }

            struct PrintPcRelOffset<T> {
                val: T,
                pc_base: u64,
            }

            impl std::fmt::Display for PrintPcRelOffset<i32> {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "0x{:x}", self.pc_base.wrapping_add((self.val + 8) as u64))
               }
            }

            impl std::fmt::Display for PrintPcRelOffset<u32> {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "0x{:x}", self.pc_base + self.val as u64 + 8)
               }
            }
        });
    }
}

struct PrintField {
    ident: Ident,
    is_modifier: bool,
    field_type: FieldType,
    print_as: PrintAs,
}

impl PrintField {
    fn new(
        field: &LoadField,
        is_modifier: bool,
        print_type: PrintAs,
    ) -> PrintField {
        PrintField {
            ident: field.ident_print(),
            is_modifier,
            field_type: field
                .field
                .field_type()
                .cloned()
                .unwrap_or(FieldType::Uint(32)),
            print_as: print_type,
        }
    }

    fn to_tokens(&self, fmt_prefix: &str, ts: &mut TokenStream) {
        let mut fmt_str = String::from(fmt_prefix);

        if self.is_modifier {
            fmt_str += ".";
        }

        fmt_str += "{}";

        let mut body_ts: TokenStream = Default::default();

        self.print_as.to_tokens(&self.ident, &mut body_ts);

        let ident = &self.ident;

        body_ts.extend(quote! {
            write!(f, #fmt_str, #ident)?;
        });

        // If the field is an enum with a "none" value, only print it if it isn't that.
        match &self.field_type {
            FieldType::Enum(enm) if enm.has_none => {
                ts.extend(quote! {
                    if !#ident.is_none() {
                        #body_ts
                    }
                });
            }
            _ => ts.extend(body_ts),
        }
    }
}

struct InstrFields<'a> {
    instr: &'a Instr,
    phy_loads: BTreeMap<String, LoadField<'a>>,
    vir_loads: Vec<LoadField<'a>>,
}

impl<'a> InstrFields<'a> {
    pub fn new(instr: &Instr) -> InstrFields<'_> {
        InstrFields {
            instr,
            phy_loads: Default::default(),
            vir_loads: Default::default(),
        }
    }

    pub fn load(&mut self, name: &str) -> &LoadField<'a> {
        let field = self
            .instr
            .get_named_field(name)
            .expect("modifier references non-existent field");

        match field {
            InstrField::Physical(f) => self
                .phy_loads
                .entry(f.name.clone())
                .or_insert(LoadField::new(f.name.as_str(), self.instr)),
            InstrField::Virtual(f) => {
                self.vir_loads
                    .push(LoadField::new(f.name.as_str(), self.instr));
                let load_idx = self.vir_loads.len() - 1;

                for used_field in &f.expr.fields() {
                    self.load(&used_field.name);
                }

                &self.vir_loads[load_idx]
            }
            InstrField::Reserved(_) => unreachable!("Can't load reserved"),
        }
    }
}

impl ToTokens for InstrFields<'_> {
    fn to_tokens(&self, ts: &mut TokenStream) {
        self.phy_loads
            .values()
            .chain(self.vir_loads.iter())
            .for_each(|l| l.to_tokens(ts));
    }
}

fn unique_instr_ident(prefix: &str, instr: &Instr) -> Ident {
    let variant = if let Some(var_name) = &instr.variant {
        format!("_{}", var_name.to_lowercase())
    } else {
        String::from("")
    };

    Ident::new(
        &format!(
            "{}_{}{}_{}_{}",
            prefix,
            instr.name.to_lowercase(),
            variant,
            instr.arch.start,
            instr.arch.end
        ),
        Span::call_site(),
    )
}

struct InstrPrint<'a> {
    instr: &'a Instr,
    fields: InstrFields<'a>,
    prints: Vec<PrintField>,
    fn_ident: Ident,
}

impl InstrPrint<'_> {
    pub fn new(instr: &Instr) -> InstrPrint<'_> {
        let mut fields = InstrFields::new(instr);
        let mut fragments: Vec<PrintField> = Default::default();

        // Collect all values needed for printing. This includes values that are
        // not directly printed, but used to compute printed virtual fields.
        for elem in &instr.syntax.elements {
            match elem {
                SyntaxElement::Name(n) => {
                    for mod_ in &n.mods {
                        let ld = fields.load(&mod_.name);
                        let pfield = PrintField::new(ld, true, PrintAs::Id);
                        fragments.push(pfield);
                    }
                }
                SyntaxElement::Staging(s) => {
                    let field = instr.get_sr_index_field(s.is_input).unwrap();
                    let print_as = PrintAsStaging::new(s.is_input, instr);
                    fields.load(&format!("{}", print_as.count_id));
                    let ld =
                        fields.load(&format!("{}", field.ident().unwrap()));
                    let pf =
                        PrintField::new(ld, false, PrintAs::Staging(print_as));
                    fragments.push(pf);
                }
                SyntaxElement::Imm(s) => {
                    let ld = fields.load(&s.name);
                    let print_as = PrintAs::Imm(s.type_.into());
                    let pf = PrintField::new(ld, false, print_as);
                    fragments.push(pf);
                }
                SyntaxElement::Dst(d) => {
                    let fname = "dest_register_index";
                    let is64 = matches!(
                        instr.variant.as_deref(),
                        Some("i64") | Some("u64") | Some("s64")
                    );
                    let ld = fields.load(fname);
                    let pf = PrintField::new(ld, false, PrintAs::Dst(is64));
                    fragments.push(pf);

                    for mod_ in &d.mods {
                        let ld = fields.load(&mod_.name);
                        let pf = PrintField::new(ld, true, PrintAs::Id);
                        fragments.push(pf);
                    }
                }
                SyntaxElement::PcRelOffset(o) => {
                    let ld = fields.load(&o.name);
                    let pf = PrintField::new(ld, false, PrintAs::PcRelOffset);
                    fragments.push(pf);
                }
                SyntaxElement::Src(s) => {
                    let field = instr.get_named_field(&s.name).unwrap();
                    let is64 =
                        matches!(field.field_type(), Some(FieldType::Source64));
                    let ld = fields.load(&s.name);
                    let pf = PrintField::new(ld, false, PrintAs::Src(is64));
                    fragments.push(pf);
                    fields.load("fau_page_index");

                    for mod_ in &s.mods {
                        let ld = fields.load(&mod_.name);
                        let pf = PrintField::new(ld, true, PrintAs::Id);
                        fragments.push(pf);
                    }
                }
            }
        }

        InstrPrint {
            instr,
            fields,
            prints: fragments,
            fn_ident: unique_instr_ident("print", instr),
        }
    }

    pub fn dispatch_case_ts(&self) -> TokenStream {
        let i = self.instr;

        let arch_first = i.arch.start;
        let arch_last = i.arch.end - 1;
        let arch_guard = quote! {#arch_first..=#arch_last};

        let ident = &self.fn_ident;
        let mn = Ident::new(&i.name, Span::call_site());
        if let Some(var_name) = &i.variant {
            let vi = Ident::new(var_name, Span::call_site());
            quote! { (M::#mn, Some(V::#vi), #arch_guard) => #ident(v, arch, f, ctx), }
        } else {
            quote! { (M::#mn, None, #arch_guard) => #ident(v, arch, f, ctx), }
        }
    }

    pub fn declare_types(ts: &mut TokenStream) {
        ts.extend(quote! {
            struct NullWriter;

            impl std::io::Write for NullWriter {
                fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                    Ok(buf.len())
                }
                fn flush(&mut self) -> std::io::Result<()> {
                    Ok(())
                }
            }

            #[derive(Default)]
            pub struct PrintCtx {
                pub fau32: bool,
                pub pc_base: u64,
            }

            enum PrintError {
                Fmt(std::fmt::Error),
                Io(std::io::Error),
                InvalidInstr(InvalidInstrError),
            }

            impl PrintError {
                fn into_invalid_instr(self) -> InvalidInstrError {
                    match self {
                        PrintError::InvalidInstr(e) => e,
                        _ => panic!("into_invalid_instr called on Fmt or Io"),
                    }
                }
            }

            impl From<std::convert::Infallible> for PrintError {
                fn from(_err: std::convert::Infallible) -> PrintError {
                    panic!("Infallible can't happen");
                }
            }

            impl From<std::fmt::Error> for PrintError {
                fn from(err: std::fmt::Error) -> PrintError {
                    PrintError::Fmt(err)
                }
            }

            impl From<std::io::Error> for PrintError {
                fn from(err: std::io::Error) -> PrintError {
                    PrintError::Io(err)
                }
            }

            impl From<EncodeError> for PrintError {
                fn from(err: EncodeError) -> PrintError {
                    PrintError::InvalidInstr(err.into())
                }
            }

            impl From<InvalidInstrError> for PrintError {
                fn from(err: InvalidInstrError) -> PrintError {
                    PrintError::InvalidInstr(err)
                }
            }
        });
    }
}

impl ToTokens for InstrPrint<'_> {
    fn to_tokens(&self, ts: &mut TokenStream) {
        let display_name = self.instr.full_name();

        let mut body_ts: TokenStream = Default::default();

        self.fields.to_tokens(&mut body_ts);

        // Name always comes first, ignore the order from the syntax elements.
        body_ts.extend(quote! {write!(f, #display_name)?;});

        // The first argument after the name and it's modifiers doesn't yet use
        // ',' as a separator.
        let mut had_non_mod = false;
        for frag in self.prints.iter() {
            let with_comma = had_non_mod && !frag.is_modifier;
            let prefix = if frag.is_modifier {
                ""
            } else if with_comma {
                ", "
            } else {
                " "
            };
            frag.to_tokens(prefix, &mut body_ts);
            had_non_mod |= !frag.is_modifier;
        }

        let ident = &self.fn_ident;
        ts.extend(quote! {
            fn #ident (
                v: u64, arch: u8, f: &mut dyn std::io::Write, ctx: &PrintCtx
            ) -> Result<(), PrintError> {
                #body_ts
                Ok(())
            }
        });
    }
}

fn gen_print(instrs: &Vec<Instr>) -> TokenStream {
    let mut ts: TokenStream = Default::default();
    let mut match_arms_ts: TokenStream = Default::default();

    InstrPrint::declare_types(&mut ts);
    PrintAs::declare_types(&mut ts);

    for instr in instrs {
        let instr_print = InstrPrint::new(instr);
        match_arms_ts.extend(instr_print.dispatch_case_ts());
        instr_print.to_tokens(&mut ts);
    }

    ts.extend(quote! {
        pub fn print(v: u64, mn: Mnemonic, var: Option<Variant>, arch: u8,
                     f: &mut impl std::io::Write, ctx: &PrintCtx
        ) -> std::io::Result<()> {
            use Mnemonic as M;
            use Variant as V;
            let res = match (&mn, var, arch) {
                #match_arms_ts
                _ => write!(f, "<invalid input>").map_err(Into::into),
            };

            match (res) {
                Err(PrintError::InvalidInstr(e)) => write!(f, "{}", e),
                Err(PrintError::Fmt(e)) => write!(f, "<format error>"),
                Err(PrintError::Io(e)) => Err(e),
                Ok(()) => Ok(()),
            }
        }
    });

    ts
}

struct SimpleEnum {
    name: String,
    values: Vec<String>,
}

impl SimpleEnum {
    pub fn new(name: &str, values: Vec<String>) -> SimpleEnum {
        SimpleEnum {
            name: String::from(name),
            values: values,
        }
    }

    pub fn ident(&self) -> Ident {
        Ident::new(&self.name, Span::call_site())
    }
}

impl ToTokens for SimpleEnum {
    fn to_tokens(&self, ts: &mut TokenStream) {
        let name_id = ident!("{}", &self.name);

        let mut idents: Vec<Ident> =
            self.values.iter().map(|n| ident!("{n}")).collect();
        idents.sort();

        let display_cases: Vec<TokenStream> = idents
            .iter()
            .zip(&self.values)
            .map(|(i, s)| quote! {#name_id::#i => write!(f, #s)})
            .collect();

        ts.extend(quote! {
            pub enum #name_id {
                #(#idents),*
            }

            impl std::fmt::Display for #name_id {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        #(#display_cases),*
                    }
                }
            }
        })
    }
}

fn gen_decode(isa: &ISA, name_e: Ident, var_e: Ident) -> TokenStream {
    let mut decode_per_arch: BTreeMap<_, Box<DecoderNode>> = Default::default();

    for target_arch in isa.arch.clone() {
        let instrs: Vec<&Instr> = isa
            .instrs
            .iter()
            .filter(|i| i.arch.contains(&target_arch))
            .collect();

        let decoder = DecoderNode::build(instrs);
        decode_per_arch.insert(target_arch, Box::new(decoder));
    }

    let make_ret_val = |instr: &Instr| {
        let name = Ident::new(&instr.name, Span::call_site());
        if let Some(variant) = &instr.variant {
            let vi = Ident::new(variant, Span::call_site());
            quote! { (M::#name, Some(V::#vi)) }
        } else {
            quote! { (M::#name, None) }
        }
    };

    let make_check = |instr: &Instr| {
        let fn_ident = unique_instr_ident("print", instr);
        quote! {(#fn_ident(input_value, arch, &mut NullWriter, &Default::default()))
        .map_err(|e| e.into_invalid_instr())}
    };

    let decoder_cases_ts: TokenStream = decode_per_arch
        .iter()
        .map(|(arch, decoder)| {
            let per_arch = decoder.to_tokens(&make_ret_val, &make_check);
            quote! {
                #arch => #per_arch,
            }
        })
        .collect();

    quote! {
        pub fn try_decode(
            input_value: u64, arch: u8
        ) -> Result<(Mnemonic, Option<Variant>), InvalidInstrError> {
            use #name_e as M;
            use #var_e as V;
            match arch {
                #decoder_cases_ts
                _ => Err(InvalidInstrError::Any),
            }
        }
    }
}

pub fn gen_decoder(
    xml_file: &str,
    arch: std::ops::Range<u8>,
) -> Result<TokenStream> {
    let isa = ISA::from_xml_file(std::fs::File::open(xml_file)?, arch)?;

    let instr_names: HashSet<String> =
        isa.instrs.iter().map(|i| i.name.clone()).collect();
    let instr_variants: HashSet<String> = isa
        .instrs
        .iter()
        .filter_map(|i| i.variant.clone())
        .collect();

    let instr_name_enum =
        SimpleEnum::new("Mnemonic", instr_names.into_iter().collect());
    let instr_var_enum =
        SimpleEnum::new("Variant", instr_variants.into_iter().collect());

    let printer_ts = gen_print(&isa.instrs);
    let decode_ts =
        gen_decode(&isa, instr_name_enum.ident(), instr_var_enum.ident());

    Ok(quote! {
        pub mod decode {
            use super::*;
            use crate::isa::*;

            pub type SourceEncoding =
                super::SourceEncoding::<SmallConstantT, FauSpecialIndexPage>;
            pub type SourceEncoding64 =
                super::SourceEncoding64::<SmallConstantT, FauSpecialIndexPage>;

            fn sign_ext(v: u32, enc_width: u8) -> i32 {
                let r = 32 - enc_width;
                ((v << r) as i32) >> r
            }

            #instr_name_enum

            #instr_var_enum

            #decode_ts

            #printer_ts
        }
    })
}
