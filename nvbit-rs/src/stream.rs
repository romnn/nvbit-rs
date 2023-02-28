#![allow(warnings)]

use serde::ser::{SerializeSeq, Serializer};

// pub struct Encoder<'a, W, Seq> {
// pub struct Encoder<W, Seq> {
// pub struct Encoder<'a, Seq> {
pub struct Encoder<Seq> {
    // pub serializer: serde_json::Serializer<W>,
    // pub seq: &'a mut Seq,
    pub seq: Seq,
}

// pub struct JSONEncoder {}

// pub struct JSONEncoder<'a, W> {
//     pub serializer: serde_json::Serializer<W>,
//     // serializer: serde_json::Serializer::new(writer),
//     pub encoder: Encoder<'a, serde_json::ser::Compound<'a, W, serde_json::ser::CompactFormatter>>,
// }

// impl<W> JSONEncoder<'_, W>
// impl JSONEncoder
// // where
// //     W: std::io::Write,
// {
//     // pub fn new<'a, Seq, W: std::io::Write>(
//     pub fn new<W: std::io::Write>(
//         writer: W,
//     ) -> serde_json::Result<
//         Encoder<serde_json::ser::Compound<'static, W, serde_json::ser::CompactFormatter>>,
//     > {
//         // pub fn new<Seq>(writer: W) -> serde_json::Result<Self> {
//         let mut serializer = serde_json::Serializer::new(writer);
//         serde_json::Result::Ok(Encoder::new(&mut serializer))
//         // let seq = &mut serializer.serialize_seq(None)?;
//         // serde_json::Result::Ok(Self {
//         //     // serializer: serde_json::Serializer::new(writer),
//         //     serializer,
//         //     encoder: Encoder { seq },
//         // })
//     }
// }

#[cfg(test)]
mod tests {
    use super::Encoder;
    use anyhow::Result;

    #[test]
    fn json() -> Result<()> {
        let mut buf = std::io::BufWriter::new(Vec::new());
        let mut serializer = serde_json::Serializer::new(&mut buf);
        let mut encoder = Encoder::new(&mut serializer);
        encoder.encode::<u32>(1)?;
        encoder.encode::<u32>(2)?;
        encoder.encode::<u32>(3)?;
        encoder.finalize()?;
        let serialized = String::from_utf8(buf.into_inner()?)?;
        assert_eq!(serialized, "[1,2,3]");
        Ok(())
    }
}

// impl<W> Encoder<W> {
// impl<'a, W, Seq> Encoder<'a, W, Seq>
// impl<W, Seq> Encoder<W, Seq>
// impl<'a, Seq> Encoder<'a, Seq>
impl<Seq> Encoder<Seq>
where
    // W: std::io
    // Seq: SerializeSeq + 'a,
    Seq: SerializeSeq,
{
    // pub fn new<S>(serializer: &mut S) -> Self
    pub fn new<S>(serializer: S) -> Self
    where
        // S: serde::Serializer<SerializeSeq = &'a mut Seq>,
        S: serde::Serializer<SerializeSeq = Seq>,
    {
        let seq = serializer.serialize_seq(None).unwrap();
        // self.seq.end()
        Self { seq }
    }

    // pub fn finalize(mut self) -> serde_json::Result<()> {
    pub fn finalize(self) -> Result<Seq::Ok, Seq::Error> {
        self.seq.end()
    }

    // // pub fn encode<V>(&mut self, value: V) -> serde_json::Result<()> {
    pub fn encode<V>(&mut self, value: impl serde::Serialize) -> Result<(), Seq::Error> {
        // serde_json::Result::Ok(())
        // let mut ser = serde_json::Serializer::new(self.writer);
        // let mut seq = ser.serialize_seq(None)?;
        // for row in rows {
        //     seq.serialize_element(&row)?;
        self.seq.serialize_element(&value)
        // }
        // seq.end()
    }
}

// fn main() ->  {
//     let rows = /* whatever iterator */ "serde".chars();
//     let out = std::io::stdout();

//     let mut ser = serde_json::Serializer::new(out);
//     let mut seq = ser.serialize_seq(None)?;
//     for row in rows {
//         seq.serialize_element(&row)?;
//     }
//     seq.end()
// }
