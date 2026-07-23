//! Tensor shape, scalar-cell, dtype, and promotion validation.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;

use sim_kernel::{Cx, DefaultFactory, Error, NoopEvalPolicy, Result, Symbol, Value};

use crate::spec::checked_element_count;

use super::domain::number_domain;

pub(super) fn validate_shape_and_data_len(shape: &[usize], data_len: usize) -> Result<()> {
    let expected = checked_element_count(shape)?;
    if data_len != expected {
        return Err(Error::Eval(format!(
            "tensor shape {:?} expects {expected} cells, found {data_len}",
            shape
        )));
    }
    Ok(())
}

pub(super) fn validate_cells(cx: &mut Cx, data: &[Value]) -> Result<()> {
    for cell in data {
        let Some(number) = cx.number_value_ref(cell.clone())? else {
            return Err(Error::Eval(
                "tensor cells must all be scalar number values".to_owned(),
            ));
        };
        if number.domain == number_domain() {
            return Err(Error::Eval(
                "tensor cells must be scalar numbers, not nested tensors".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_dtype_accepts_cells(
    cx: &mut Cx,
    dtype: &Symbol,
    data: &[Value],
) -> Result<()> {
    let domains = cell_domains(cx, data)?;
    if domains
        .iter()
        .all(|domain| promotion_cost(cx, domain, dtype).is_some())
    {
        return Ok(());
    }
    Err(Error::Eval(format!(
        "tensor dtype {dtype} is not a valid join for cell domains {domains:?}"
    )))
}

pub(super) fn validate_exact_cell_dtype(dtype: &Symbol, data: &[Value]) -> Result<()> {
    let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    for cell in data {
        let Some(number) = cell.object().as_number_value() else {
            return Err(Error::Eval(
                "tensor cells must all be scalar number values".to_owned(),
            ));
        };
        let domain = number.number_domain(&mut cx)?;
        if domain == number_domain() {
            return Err(Error::Eval(
                "tensor cells must be scalar numbers, not nested tensors".to_owned(),
            ));
        }
        if &domain != dtype {
            return Err(Error::Eval(format!(
                "tensor dtype {dtype} does not match cell domain {domain}"
            )));
        }
    }
    Ok(())
}

pub(super) fn choose_dtype(
    cx: &mut Cx,
    dtype_hint: Option<Symbol>,
    data: &[Value],
) -> Result<Symbol> {
    if data.is_empty() {
        return dtype_hint
            .filter(|dtype| dtype != &number_domain())
            .ok_or_else(|| {
                Error::Eval("an empty tensor requires an explicit scalar dtype".to_owned())
            });
    }
    let domains = cell_domains(cx, data)?;
    if let Some(dtype) = dtype_hint {
        if domains
            .iter()
            .all(|domain| promotion_cost(cx, domain, &dtype).is_some())
        {
            return Ok(dtype);
        }
        return Err(Error::Eval(format!(
            "tensor dtype {dtype} is not a valid join for cell domains {domains:?}"
        )));
    }
    let candidates = cx
        .registry()
        .number_domains()
        .keys()
        .filter(|symbol| **symbol != number_domain())
        .cloned()
        .collect::<Vec<_>>();
    let mut best = None::<(u32, Symbol)>;
    for candidate in candidates {
        let mut total = 0u32;
        let mut valid = true;
        for domain in &domains {
            let Some(cost) = promotion_cost(cx, domain, &candidate) else {
                valid = false;
                break;
            };
            total += cost;
        }
        if !valid {
            continue;
        }
        match &best {
            Some((best_cost, best_symbol))
                if total > *best_cost || (total == *best_cost && candidate >= *best_symbol) => {}
            _ => best = Some((total, candidate)),
        }
    }
    best.map(|(_, symbol)| symbol).ok_or_else(|| {
        Error::Eval(format!(
            "no join domain exists for tensor cells {domains:?}"
        ))
    })
}

fn cell_domains(cx: &mut Cx, data: &[Value]) -> Result<Vec<Symbol>> {
    data.iter()
        .map(|value| {
            cx.number_value_ref(value.clone())?
                .map(|number| number.domain)
                .ok_or_else(|| {
                    Error::Eval("tensor cells must all be scalar number values".to_owned())
                })
        })
        .collect()
}

fn promotion_cost(cx: &Cx, from: &Symbol, to: &Symbol) -> Option<u32> {
    if from == to {
        return Some(0);
    }

    #[derive(Clone, Eq, PartialEq)]
    struct State {
        cost: u32,
        symbol: Symbol,
    }

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .cost
                .cmp(&self.cost)
                .then_with(|| other.symbol.cmp(&self.symbol))
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut best = BTreeMap::<Symbol, u32>::new();
    let mut heap = BinaryHeap::new();
    best.insert(from.clone(), 0);
    heap.push(State {
        cost: 0,
        symbol: from.clone(),
    });

    while let Some(State { cost, symbol }) = heap.pop() {
        if &symbol == to {
            return Some(cost);
        }
        if best.get(&symbol).copied().unwrap_or(u32::MAX) < cost {
            continue;
        }
        for rule in cx
            .registry()
            .value_promotion_rules()
            .iter()
            .filter(|rule| rule.from_domain == symbol)
        {
            let next = cost + rule.cost as u32;
            let entry = best.entry(rule.to_domain.clone()).or_insert(u32::MAX);
            if next < *entry {
                *entry = next;
                heap.push(State {
                    cost: next,
                    symbol: rule.to_domain.clone(),
                });
            }
        }
    }
    None
}
