use serde::ser::SerializeSeq;

/// Trait for all types that wrap raw pointers.
pub trait AsPtr<T> {
    fn as_ptr(&self) -> *const T;
}

impl<T> AsPtr<T> for *mut T {
    fn as_ptr(&self) -> *const T {
        *self as _
    }
}

impl<T> AsPtr<T> for *const T {
    fn as_ptr(&self) -> *const T {
        self.cast()
    }
}

/// Serializes a types wrapping a pointer.
///
/// # Errors
/// If the value cannot be serialized.
pub fn to_raw_ptr<S, P, T>(x: &P, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    P: AsPtr<T>,
{
    s.serialize_u64(x.as_ptr() as u64)
}

/// An encoder to serialize a stream of packets.
#[derive(Debug, Clone)]
pub struct Encoder<Seq> {
    pub seq: Seq,
}

impl<Seq> Encoder<Seq>
where
    Seq: SerializeSeq,
{
    /// Creates a new encoder for a `serde::Serializer`.
    ///
    /// # Errors
    /// If the serializer cannot start a new sequence.
    pub fn new<S>(serializer: S) -> Result<Self, S::Error>
    where
        S: serde::Serializer<SerializeSeq = Seq>,
    {
        let seq = serializer.serialize_seq(None)?;
        Ok(Self { seq })
    }

    /// Finalizes the sequence.
    ///
    /// This will terminate the sequence and consume the encoder.
    ///
    /// # Errors
    /// If there is no sequence to terminate (this should never happen).
    pub fn finalize(self) -> Result<Seq::Ok, Seq::Error> {
        self.seq.end()
    }

    /// Encode a single value as a sequence element.
    ///
    /// # Errors
    /// If the element cannot be serialized using `SerializeSeq`,
    /// an error is returned.
    pub fn encode<V>(&mut self, value: impl serde::Serialize) -> Result<(), Seq::Error> {
        self.seq.serialize_element(&value)
    }
}

/// A decoder to deserialize a stream of packets.
#[derive(Debug, Clone)]
pub struct Decoder<T, CB> {
    callback: CB,
    phantom: std::marker::PhantomData<T>,
}

impl<T, CB> Decoder<T, CB> {
    pub fn new(callback: CB) -> Self {
        Self {
            callback,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'de, T, CB> serde::de::Visitor<'de> for Decoder<T, CB>
where
    T: serde::Deserialize<'de>,
    CB: FnMut(T),
{
    type Value = ();

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("array of packets")
    }

    fn visit_seq<A>(mut self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        while let Some(item) = seq.next_element::<T>()? {
            (self.callback)(item);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Decoder, Encoder};
    use anyhow::Result;
    use serde::Deserializer;
    use std::io::{BufReader, BufWriter, Cursor};

    #[test]
    fn json() -> Result<()> {
        let mut writer = BufWriter::new(Vec::new());
        let mut serializer = serde_json::Serializer::new(&mut writer);
        let mut encoder = Encoder::new(&mut serializer)?;
        encoder.encode::<u32>(1)?;
        encoder.encode::<u32>(2)?;
        encoder.encode::<u32>(3)?;
        encoder.finalize()?;

        let buf = writer.into_inner()?;
        let result = std::str::from_utf8(&buf)?;
        assert_eq!(result, "[1,2,3]");

        let mut values: Vec<u32> = Vec::new();
        let reader = BufReader::new(Cursor::new(buf));
        let mut deserializer = serde_json::Deserializer::from_reader(reader);
        let decoder = Decoder::new(|value| values.push(value));
        deserializer.deserialize_seq(decoder)?;

        assert_eq!(values, [1, 2, 3]);
        Ok(())
    }

    #[test]
    fn messagepack() -> Result<()> {
        let buf = Vec::new();
        let mut writer = BufWriter::new(buf);
        let mut serializer = rmp_serde::Serializer::new(&mut writer);
        let mut encoder = Encoder::new(&mut serializer)?;
        encoder.encode::<u32>(1)?;
        encoder.encode::<u32>(2)?;
        encoder.encode::<u32>(3)?;
        encoder.finalize()?;

        let mut values: Vec<u32> = Vec::new();
        let reader = BufReader::new(Cursor::new(writer.into_inner()?));
        let mut deserializer = rmp_serde::Deserializer::new(reader);
        let decoder = Decoder::new(|value| values.push(value));
        deserializer.deserialize_seq(decoder)?;

        assert_eq!(values, [1, 2, 3]);
        Ok(())
    }
}
