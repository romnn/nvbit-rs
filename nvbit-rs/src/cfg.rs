use super::Instruction;

#[derive(Debug, Default)]
pub struct BasicBlock<'a> {
    pub instructions: Vec<&'a Instruction<'a>>,
}

#[derive(Debug, Default)]
pub struct CFG<'a> {
    pub is_degenerate: bool,
    pub basic_blocks: Vec<BasicBlock<'a>>,
}
