#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tag {
    None = 0,
    Tensor = 1,
    String = 2,
    Double = 3,
    Int = 4,
    Bool = 5,
    ListBool = 6,
    ListDouble = 7,
    ListInt = 8,
    ListTensor = 9,
    ListScalar = 10,
    ListOptionalTensor = 11,
}

impl TryFrom<u32> for Tag {
    type Error = ();

    fn try_from(val: u32) -> core::result::Result<Self, ()> {
        match val {
            0 => Ok(Self::None),
            1 => Ok(Self::Tensor),
            2 => Ok(Self::String),
            3 => Ok(Self::Double),
            4 => Ok(Self::Int),
            5 => Ok(Self::Bool),
            6 => Ok(Self::ListBool),
            7 => Ok(Self::ListDouble),
            8 => Ok(Self::ListInt),
            9 => Ok(Self::ListTensor),
            10 => Ok(Self::ListScalar),
            11 => Ok(Self::ListOptionalTensor),
            _ => Err(()),
        }
    }
}
