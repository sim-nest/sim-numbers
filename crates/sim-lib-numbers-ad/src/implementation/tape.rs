//! Reverse-mode autodiff tape: a graph of recorded operations with forward
//! evaluation and reverse gradient accumulation.

/// A handle to a node recorded on a [`Tape`]: its index into the tape.
///
/// Returned by every tape operation and passed back in to build further
/// operations or to read a value or gradient.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Var(pub usize);

/// One recorded operation on a [`Tape`].
///
/// Operands and arguments are stored as the indices ([`Var`] positions) of the
/// nodes they consume, forming the computation graph that the reverse pass
/// walks to accumulate gradients.
#[derive(Clone, Debug, PartialEq)]
pub enum TapeNode {
    /// A constant with the given value.
    Const(f64),
    /// An independent input bound to the given gradient slot.
    Input(usize),
    /// Sum of the two operand nodes.
    Add(usize, usize),
    /// Difference of the two operand nodes (first minus second).
    Sub(usize, usize),
    /// Product of the two operand nodes.
    Mul(usize, usize),
    /// Quotient of the two operand nodes (first divided by second).
    Div(usize, usize),
    /// Sine of the operand node.
    Sin(usize),
    /// Cosine of the operand node.
    Cos(usize),
    /// Exponential of the operand node.
    Exp(usize),
    /// Natural logarithm of the operand node.
    Ln(usize),
    /// Square root of the operand node.
    Sqrt(usize),
    /// Reciprocal of the operand node.
    Recip(usize),
}

/// A reverse-mode autodiff tape: the recorded operation graph plus the forward
/// value of each node.
///
/// Build an expression by calling the operation methods, which append nodes and
/// return [`Var`] handles; the forward value is computed eagerly as each node
/// is pushed. Call [`grad`](Tape::grad) on an output node to run the reverse
/// pass and accumulate the partial derivatives with respect to every input.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_ad::Tape;
///
/// // f(a, b) = a * b + a, at a = 2, b = 5.
/// let mut tape = Tape::new();
/// let a = tape.input(0, 2.0);
/// let b = tape.input(1, 5.0);
/// let product = tape.mul(a, b);
/// let out = tape.add(product, a);
/// assert_eq!(tape.value(out), 12.0);
/// // df/da = b + 1 = 6, df/db = a = 2.
/// assert_eq!(tape.grad(out, 2), vec![6.0, 2.0]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Tape {
    nodes: Vec<TapeNode>,
    values: Vec<f64>,
}

impl Tape {
    /// Creates an empty tape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a constant node and returns its handle.
    pub fn constant(&mut self, value: f64) -> Var {
        self.push(TapeNode::Const(value), value)
    }

    /// Records an independent input bound to gradient slot `slot` with the given
    /// value, and returns its handle.
    pub fn input(&mut self, slot: usize, value: f64) -> Var {
        self.push(TapeNode::Input(slot), value)
    }

    /// Records `a + b` and returns the handle of the sum.
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.push(TapeNode::Add(a.0, b.0), self.values[a.0] + self.values[b.0])
    }

    /// Records `a - b` and returns the handle of the difference.
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        self.push(TapeNode::Sub(a.0, b.0), self.values[a.0] - self.values[b.0])
    }

    /// Records `a * b` and returns the handle of the product.
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        self.push(TapeNode::Mul(a.0, b.0), self.values[a.0] * self.values[b.0])
    }

    /// Records `a / b` and returns the handle of the quotient.
    pub fn div(&mut self, a: Var, b: Var) -> Var {
        self.push(TapeNode::Div(a.0, b.0), self.values[a.0] / self.values[b.0])
    }

    /// Records `sin(arg)` and returns its handle.
    pub fn sin(&mut self, arg: Var) -> Var {
        self.push(TapeNode::Sin(arg.0), self.values[arg.0].sin())
    }

    /// Records `cos(arg)` and returns its handle.
    pub fn cos(&mut self, arg: Var) -> Var {
        self.push(TapeNode::Cos(arg.0), self.values[arg.0].cos())
    }

    /// Records `exp(arg)` and returns its handle.
    pub fn exp(&mut self, arg: Var) -> Var {
        self.push(TapeNode::Exp(arg.0), self.values[arg.0].exp())
    }

    /// Records `ln(arg)` and returns its handle.
    pub fn ln(&mut self, arg: Var) -> Var {
        self.push(TapeNode::Ln(arg.0), self.values[arg.0].ln())
    }

    /// Records `sqrt(arg)` and returns its handle.
    pub fn sqrt(&mut self, arg: Var) -> Var {
        self.push(TapeNode::Sqrt(arg.0), self.values[arg.0].sqrt())
    }

    /// Records `1 / arg` and returns its handle.
    pub fn recip(&mut self, arg: Var) -> Var {
        self.push(TapeNode::Recip(arg.0), self.values[arg.0].recip())
    }

    /// Returns the forward value recorded for `var`.
    pub fn value(&self, var: Var) -> f64 {
        self.values[var.0]
    }

    /// Runs the reverse pass from output `out` and returns the gradient with
    /// respect to the `n_inputs` input slots.
    ///
    /// Seeds the adjoint of `out` with `1.0`, walks the recorded nodes in
    /// reverse, and accumulates each input slot's partial derivative; the
    /// returned vector has length `n_inputs`.
    pub fn grad(&self, out: Var, n_inputs: usize) -> Vec<f64> {
        let mut adjoints = vec![0.0; self.nodes.len()];
        let mut input_grad = vec![0.0; n_inputs];
        adjoints[out.0] = 1.0;

        for index in (0..self.nodes.len()).rev() {
            let seed = adjoints[index];
            if seed == 0.0 {
                continue;
            }
            match self.nodes[index] {
                TapeNode::Const(_) => {}
                TapeNode::Input(slot) => {
                    if let Some(grad) = input_grad.get_mut(slot) {
                        *grad += seed;
                    }
                }
                TapeNode::Add(a, b) => {
                    adjoints[a] += seed;
                    adjoints[b] += seed;
                }
                TapeNode::Sub(a, b) => {
                    adjoints[a] += seed;
                    adjoints[b] -= seed;
                }
                TapeNode::Mul(a, b) => {
                    adjoints[a] += seed * self.values[b];
                    adjoints[b] += seed * self.values[a];
                }
                TapeNode::Div(a, b) => {
                    let denom = self.values[b] * self.values[b];
                    adjoints[a] += seed / self.values[b];
                    adjoints[b] -= seed * self.values[a] / denom;
                }
                TapeNode::Sin(arg) => adjoints[arg] += seed * self.values[arg].cos(),
                TapeNode::Cos(arg) => adjoints[arg] -= seed * self.values[arg].sin(),
                TapeNode::Exp(arg) => adjoints[arg] += seed * self.values[index],
                TapeNode::Ln(arg) => adjoints[arg] += seed / self.values[arg],
                TapeNode::Sqrt(arg) => adjoints[arg] += seed / (2.0 * self.values[index]),
                TapeNode::Recip(arg) => {
                    let denom = self.values[arg] * self.values[arg];
                    adjoints[arg] -= seed / denom;
                }
            }
        }

        input_grad
    }

    fn push(&mut self, node: TapeNode, value: f64) -> Var {
        let index = self.nodes.len();
        self.nodes.push(node);
        self.values.push(value);
        Var(index)
    }
}
