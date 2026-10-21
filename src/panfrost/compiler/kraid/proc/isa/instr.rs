// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use crate::isa::xml::XmlElement;
use crate::isa::*;
use proc_macro2::{Ident, Span};
use std::rc::Rc;

impl XmlElement {
    fn get_bit_range(&self) -> Result<Range<u8>> {
        let pos = self
            .get_u8_attr("pos")
            .ok_or(err("Instruction field has no pos attribute"))?;
        let width = self
            .get_u8_attr("width")
            .ok_or(err("Instruction field has no width attribute"))?;
        Ok(pos..(pos + width))
    }
}

#[derive(Clone)]
pub enum FieldType {
    Enum(Rc<Enum>),
    PcRelOffsetSigned(u8),
    PcRelOffsetUnsigned(u8),
    Source,
    Source64,
    Int(u8),
    Uint(u8),
}

impl FieldType {
    fn from_name(type_: &str, bits: u8, enums: &EnumSet) -> Result<FieldType> {
        if let Some(e) = enums.get_enum(type_) {
            return Ok(FieldType::Enum(e.clone()));
        }

        match type_ {
            "pc_rel_label_signed" => Ok(FieldType::PcRelOffsetSigned(bits)),
            "pc_rel_label_unsigned" => Ok(FieldType::PcRelOffsetUnsigned(bits)),
            "SourceEncoding" => Ok(FieldType::Source),
            "SourceEncoding64" => Ok(FieldType::Source64),
            "float" | "V2F16" | "V2I16" | "V4I8" => {
                if bits != 32 {
                    return Err(err(
                        "Vector and float immediates should be 32-bit",
                    ));
                }
                Ok(FieldType::Uint(32))
            }
            "int" => Ok(FieldType::Int(bits)),
            "uint" => Ok(FieldType::Uint(bits)),
            _ => Err(err("Unknown field type")),
        }
    }

    pub fn is_enum(&self, name: &str) -> bool {
        if let FieldType::Enum(e) = self {
            e.name == name
        } else {
            false
        }
    }
}

pub struct FieldRestrict {
    pub values: Vec<EnumLiteral>,
}

impl FieldRestrict {
    fn from_xml_attr(
        attr: &str,
        type_: Option<&FieldType>,
    ) -> Result<Rc<FieldRestrict>> {
        let Some(FieldType::Enum(e)) = type_ else {
            return Err(err("restrict= requires an enum type"));
        };

        let mut values = Vec::new();
        for v_name in attr.split(' ') {
            if v_name.trim().is_empty() {
                continue;
            }

            let Some(v) = e.get_value(v_name) else {
                return Err(err("Invalid enum value in restrict"));
            };
            values.push(EnumLiteral::new(e, v));
        }
        Ok(Rc::new(FieldRestrict { values }))
    }
}

impl ToTokens for FieldRestrict {
    fn to_tokens(&self, ts: &mut TokenStream2) {
        let mut values_ts = TokenStream2::new();
        for i in &self.values {
            values_ts.extend(quote! { #i, });
        }
        ts.extend(quote! { [#values_ts] });
    }
}

pub struct VirtualField {
    pub name: String,
    pub ident: Ident,
    pub type_: FieldType,
    pub restrict: Option<Rc<FieldRestrict>>,
    pub expr: Box<Expr>,
}

impl VirtualField {
    fn from_xml(
        xml: XmlElement,
        _arch: Range<u8>,
        enums: &EnumSet,
    ) -> Result<VirtualField> {
        assert_eq!(xml.name.local_name, "virtual");

        let name = xml
            .attrs
            .get("name")
            .ok_or(err("Instruction virtual has no name"))?
            .to_string();
        let snake_name = to_snake_case(&name);
        let ident = Ident::new(&snake_name, Span::call_site());

        let type_name = xml
            .attrs
            .get("type")
            .ok_or(err("Instruction virtual has no type"))?;
        let type_ = FieldType::from_name(type_name, 32, enums)?;

        let restrict = if let Some(res_attr) = xml.attrs.get("restrict") {
            Some(FieldRestrict::from_xml_attr(res_attr, Some(&type_))?)
        } else {
            None
        };

        let expr = if let Some(exact) = xml.attrs.get("exact") {
            Box::new(Expr::literal(Some(type_name), exact, enums)?)
        } else {
            let mut children = xml.children;
            if children.len() != 1 {
                return Err(err("Virtual fields must have a single expr"));
            }
            let child = children.pop().unwrap();
            Box::new(Expr::from_xml(child, enums)?)
        };

        Ok(VirtualField {
            name,
            ident,
            type_,
            restrict,
            expr,
        })
    }
}

pub enum FieldMod {
    None,
    Align(u8),
    Shr(u8),
    Minus(u8),
}

impl FieldMod {
    fn args_as_u8(a: &str) -> Result<u8> {
        let a = a.strip_prefix("(").ok_or(err("Invalid modifier"))?;
        let a = a.strip_suffix(")").ok_or(err("Invalid modifier"))?;
        u8::from_str_radix(a, 10).map_err(|_| err("Invalid modifier"))
    }

    fn from_attr(attr: &str) -> Result<FieldMod> {
        if let Some(a) = attr.strip_prefix("align") {
            Ok(FieldMod::Align(FieldMod::args_as_u8(a)?))
        } else if let Some(a) = attr.strip_prefix("minus") {
            Ok(FieldMod::Minus(FieldMod::args_as_u8(a)?))
        } else if let Some(a) = attr.strip_prefix("shr") {
            Ok(FieldMod::Shr(FieldMod::args_as_u8(a)?))
        } else {
            Err(err("Invalid modifier"))
        }
    }

    pub fn extra_bits(&self) -> u8 {
        match self {
            FieldMod::None | FieldMod::Align(_) => 0,
            FieldMod::Minus(_) => 1,
            FieldMod::Shr(n) => *n,
        }
    }
}

pub struct PhysicalField {
    pub name: String,
    pub ident: Ident,
    pub bits: Range<u8>,
    pub mod_: FieldMod,
    pub type_: Option<FieldType>,
    pub restrict: Option<Rc<FieldRestrict>>,
    pub expr: Option<Box<Expr>>,
    pub exact: Option<u64>,
}

fn resolve_exact(
    type_: Option<&str>,
    value: &str,
    enums: &EnumSet,
) -> Result<u64> {
    if let Some(v) = type_
        .and_then(|t| enums.get_enum(t))
        .and_then(|e| e.get_value(value))
    {
        return Ok(v.value as u64);
    }

    u64::from_str_radix(value, 10).map_err(|_| "Unknown exact literal".into())
}

impl PhysicalField {
    fn from_xml(
        xml: XmlElement,
        _arch: Range<u8>,
        enums: &EnumSet,
    ) -> Result<PhysicalField> {
        assert_eq!(xml.name.local_name, "field");

        let name = xml
            .attrs
            .get("name")
            .ok_or(err("Instruction virtual has no name"))?
            .to_string();
        let snake_name = to_snake_case(&name);
        let ident = Ident::new(&snake_name, Span::call_site());
        let bits = xml.get_bit_range()?;

        let mod_ = if let Some(attr) = xml.attrs.get("modifier") {
            FieldMod::from_attr(attr)?
        } else {
            FieldMod::None
        };

        let type_name = xml.attrs.get("type");
        let type_ = match type_name {
            Some(name) => Some(FieldType::from_name(
                name,
                u8::try_from(bits.len()).unwrap() + mod_.extra_bits(),
                enums,
            )?),
            None => None,
        };

        let restrict = if let Some(res_attr) = xml.attrs.get("restrict") {
            Some(FieldRestrict::from_xml_attr(res_attr, type_.as_ref())?)
        } else {
            None
        };

        let mut expr = if let Some(exact) = xml.attrs.get("exact") {
            Some(Box::new(Expr::literal(
                type_name.map(String::as_str),
                exact,
                enums,
            )?))
        } else {
            None
        };

        // Store the resolved exact value if it exists for convenience.
        let exact = if let Some(exact) = xml.attrs.get("exact") {
            resolve_exact(type_name.map(String::as_str), exact, enums).ok()
        } else {
            None
        };

        for child in xml.children {
            match child.name.local_name.as_str() {
                "expression" => {
                    let e = Box::new(Expr::from_xml(child, enums)?);
                    if expr.replace(e).is_some() {
                        return Err(err(
                            "Instruction field has multiple expressions",
                        ));
                    }
                }
                _ => (),
            }
        }

        if type_.is_none() && expr.is_none() {
            return Err(err(
                "Field must have at least one of type, exact, or expression",
            ));
        }

        Ok(PhysicalField {
            name,
            ident,
            bits,
            mod_,
            type_,
            restrict,
            expr,
            exact,
        })
    }
}

pub struct ReservedField {
    pub bits: Range<u8>,
    pub zero: bool,
}

impl ReservedField {
    fn from_xml(xml: XmlElement, _arch: Range<u8>) -> Result<ReservedField> {
        assert_eq!(xml.name.local_name, "reserved");

        let type_ = xml
            .attrs
            .get("type")
            .ok_or(err("Instruction virtual has no type"))?;
        let zero = match type_.as_str() {
            "zero" => true,
            "ignore" => false,
            _ => return Err(err("Unknown reserved bits type")),
        };

        Ok(ReservedField {
            bits: xml.get_bit_range()?,
            zero,
        })
    }
}

pub enum InstrField {
    Virtual(VirtualField),
    Physical(PhysicalField),
    Reserved(ReservedField),
}

impl InstrField {
    pub fn ident(&self) -> Option<&Ident> {
        match self {
            InstrField::Virtual(f) => Some(&f.ident),
            InstrField::Physical(f) => Some(&f.ident),
            _ => None,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            InstrField::Virtual(f) => Some(&f.name),
            InstrField::Physical(f) => Some(&f.name),
            _ => None,
        }
    }

    pub fn field_type(&self) -> Option<&FieldType> {
        match self {
            InstrField::Virtual(f) => Some(&f.type_),
            InstrField::Physical(f) => f.type_.as_ref(),
            _ => None,
        }
    }

    pub fn restrict(&self) -> Option<&Rc<FieldRestrict>> {
        match self {
            InstrField::Physical(f) => f.restrict.as_ref(),
            InstrField::Virtual(f) => f.restrict.as_ref(),
            _ => None,
        }
    }
}

pub struct SyntaxModifier {
    pub name: String,
}

impl SyntaxModifier {
    fn from_xml(xml: XmlElement) -> Result<SyntaxModifier> {
        if let Some(name) = xml.attrs.get("name") {
            Ok(SyntaxModifier { name: name.clone() })
        } else {
            Err(err("modifier elements must have a name attribute"))
        }
    }
}

pub struct SyntaxName {
    pub mods: Vec<SyntaxModifier>,
}

impl SyntaxName {
    fn from_xml(xml: XmlElement) -> Result<SyntaxName> {
        let mut mods: Vec<SyntaxModifier> = Default::default();

        for child in xml.children.into_iter() {
            match child.name.local_name.as_str() {
                "modifier" => mods.push(SyntaxModifier::from_xml(child)?),
                _ => (),
            }
        }

        Ok(SyntaxName { mods })
    }
}

pub struct SyntaxStaging {
    pub is_input: bool,
}

impl SyntaxStaging {
    fn from_xml(xml: XmlElement) -> Result<SyntaxStaging> {
        if let Some(is_input) = xml.attrs.get("direction").map(|s| s == "input")
        {
            Ok(SyntaxStaging { is_input })
        } else {
            Err(err("Missing direction on staging syntax element"))
        }
    }
}

pub struct SyntaxSrc {
    pub name: String,
    pub mods: Vec<SyntaxModifier>,
}

impl SyntaxSrc {
    fn from_xml(xml: XmlElement) -> Result<SyntaxSrc> {
        let mut mods: Vec<SyntaxModifier> = Default::default();

        for child in xml.children.into_iter() {
            match child.name.local_name.as_str() {
                "modifier" => mods.push(SyntaxModifier::from_xml(child)?),
                _ => (),
            }
        }

        if let Some(name) = xml.attrs.get("name").cloned() {
            Ok(SyntaxSrc { name, mods })
        } else {
            Err(err("Missing name on src syntax element"))
        }
    }
}

pub struct SyntaxDst {
    pub mods: Vec<SyntaxModifier>,
}

impl SyntaxDst {
    fn from_xml(xml: XmlElement) -> Result<SyntaxDst> {
        let mut mods: Vec<SyntaxModifier> = Default::default();

        for child in xml.children.into_iter() {
            match child.name.local_name.as_str() {
                "modifier" => mods.push(SyntaxModifier::from_xml(child)?),
                _ => (),
            }
        }
        Ok(SyntaxDst { mods })
    }
}

#[derive(Clone, Copy)]
pub enum ImmType {
    Uint,
    Int,
    Float,
    V2F16,
    V2I16,
    V4I8,
}

pub struct SyntaxImm {
    pub name: String,
    pub type_: ImmType,
}

impl SyntaxImm {
    fn from_xml(xml: XmlElement) -> Result<SyntaxImm> {
        let elem_type = xml.name.local_name.as_str();
        let type_ = match elem_type {
            "uint" => ImmType::Uint,
            "int" => ImmType::Int,
            "float" => ImmType::Float,
            "V2I16" => ImmType::V2I16,
            "V2F16" => ImmType::V2F16,
            "V4I8" => ImmType::V4I8,
            _ => return Err("Unknown immediate type".into()),
        };

        if let Some(name) = xml.attrs.get("name").cloned() {
            Ok(SyntaxImm { name, type_ })
        } else {
            Err(err("Missing name on int syntax element"))
        }
    }
}

pub struct SyntaxPcRelOffset {
    pub name: String,
    pub signed: bool,
}

impl SyntaxPcRelOffset {
    fn from_xml(xml: XmlElement) -> Result<SyntaxPcRelOffset> {
        let signed = xml.name.local_name.as_str() == "pc_rel_label_signed";
        if let Some(name) = xml.attrs.get("name").cloned() {
            Ok(SyntaxPcRelOffset { name, signed })
        } else {
            Err(err("Missing name on src syntax element"))
        }
    }
}

pub enum SyntaxElement {
    Name(SyntaxName),
    Staging(SyntaxStaging),
    Src(SyntaxSrc),
    Dst(SyntaxDst),
    Imm(SyntaxImm),
    PcRelOffset(SyntaxPcRelOffset),
}

#[derive(Default)]
pub struct Syntax {
    pub elements: Vec<SyntaxElement>,
}

impl Syntax {
    fn from_xml(xml: XmlElement, _arch: Range<u8>) -> Result<Syntax> {
        let mut elements: Vec<SyntaxElement> = Default::default();

        for child in xml.children.into_iter() {
            match child.name.local_name.as_str() {
                "name" => elements
                    .push(SyntaxElement::Name(SyntaxName::from_xml(child)?)),
                "staging" => elements.push(SyntaxElement::Staging(
                    SyntaxStaging::from_xml(child)?,
                )),
                "src" => elements
                    .push(SyntaxElement::Src(SyntaxSrc::from_xml(child)?)),
                "dest" => elements
                    .push(SyntaxElement::Dst(SyntaxDst::from_xml(child)?)),
                "pc_rel_label_signed" | "pc_rel_label_unsigned" => elements
                    .push(SyntaxElement::PcRelOffset(
                        SyntaxPcRelOffset::from_xml(child)?,
                    )),
                _ => elements
                    .push(SyntaxElement::Imm(SyntaxImm::from_xml(child)?)),
            }
        }

        Ok(Syntax { elements })
    }
}

pub struct Instr {
    pub name: String,
    pub arch: Range<u8>,
    pub exec_unit: String,
    pub exec_time: u8,
    pub variant: Option<String>,
    pub fields: Vec<InstrField>,
    pub total_bits: u8,
    pub syntax: Syntax,
}

pub fn instr_field_ident(name: &str) -> Ident {
    let snake_name = to_snake_case(name);
    Ident::new(&snake_name, Span::call_site())
}

impl Instr {
    pub(super) fn from_xml(
        xml: XmlElement,
        arch: Range<u8>,
        enums: &EnumSet,
    ) -> Result<Instr> {
        assert_eq!(xml.name.local_name, "instruction");

        let name = xml
            .attrs
            .get("name")
            .ok_or(err("Enum has no name"))?
            .to_string();

        let exec_time = xml
            .attrs
            .get("exec_time")
            .map(|t| t.parse())
            .transpose()
            .map_err(|_| err("Could not parse exec_time"))?
            .unwrap_or(1);

        let mut i = Instr {
            name,
            arch: xml.get_arch(arch.clone()),
            exec_unit: xml
                .attrs
                .get("exec_unit")
                .ok_or(err("Instruction has no exec_unit"))?
                .to_string(),
            exec_time,
            variant: xml.attrs.get("variant").cloned(),
            fields: Default::default(),
            total_bits: 0,
            syntax: Default::default(),
        };

        for child in xml.children.into_iter() {
            let arch = i.arch.clone();
            match child.name.local_name.as_str() {
                "field" => {
                    let f = PhysicalField::from_xml(child, arch, enums)?;
                    i.total_bits = i.total_bits.max(f.bits.end);
                    i.fields.push(InstrField::Physical(f));
                }
                "virtual" => {
                    let f = VirtualField::from_xml(child, arch, enums)?;
                    i.fields.push(InstrField::Virtual(f));
                }
                "reserved" => {
                    let f = ReservedField::from_xml(child, arch)?;
                    i.total_bits = i.total_bits.max(f.bits.end);
                    i.fields.push(InstrField::Reserved(f));
                }
                "syntax" => i.syntax = Syntax::from_xml(child, arch)?,
                _ => (),
            }
        }

        Ok(i)
    }

    pub fn get_named_field(&self, name: &str) -> Option<&InstrField> {
        for f in &self.fields {
            let f_name = match f {
                InstrField::Virtual(f) => &f.name,
                InstrField::Physical(f) => &f.name,
                InstrField::Reserved(_) => continue,
            };
            if f_name == name {
                return Some(f);
            }
        }
        None
    }

    pub fn has_field_named(&self, name: &str) -> bool {
        self.get_named_field(name).is_some()
    }

    /// WARNING: This is only correct when requesting fields that exist!
    pub fn get_sr_index_field(&self, input: bool) -> Option<&InstrField> {
        let write_sr = self.get_named_field("staging_register_write_index");
        let any_sr = self.get_named_field("staging_register_index");

        // If the instruction reads and writes, write is in the _write_ version.
        // Otherwise there's only the non-write one for both use-cases.
        if !input && write_sr.is_some() {
            write_sr
        } else {
            any_sr
        }
    }

    /// WARNING: This is only correct when requesting fields that exist!
    pub fn get_sr_count_field(&self, input: bool) -> Option<&InstrField> {
        // There are some which don't follow the general mapping.
        let write_count_name = if self.name.starts_with("VAR_TEX") {
            Some("sr_write_count")
        } else {
            let sr_name =
                self.get_sr_index_field(input).and_then(InstrField::name);
            match sr_name {
                Some("staging_register_index") => Some("sr_count"),
                Some("staging_register_write_index") => Some("sr_write_count"),
                _ => None,
            }
        };

        write_count_name.and_then(|n| self.get_named_field(n))
    }

    pub fn full_name(&self) -> String {
        if let Some(variant) = &self.variant {
            format!("{}.{}", &self.name, variant)
        } else {
            self.name.clone()
        }
    }
}
