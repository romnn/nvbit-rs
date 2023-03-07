use serde::ser::SerializeSeq;

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

#[cfg(test)]
mod tests {
    use super::Encoder;
    use anyhow::Result;

    #[test]
    fn json() -> Result<()> {
        let mut buf = std::io::BufWriter::new(Vec::new());
        let mut serializer = serde_json::Serializer::new(&mut buf);
        let mut encoder = Encoder::new(&mut serializer)?;
        encoder.encode::<u32>(1)?;
        encoder.encode::<u32>(2)?;
        encoder.encode::<u32>(3)?;
        encoder.finalize()?;
        let result = String::from_utf8(buf.into_inner()?)?;
        assert_eq!(result, "[1,2,3]");
        Ok(())
    }
}
