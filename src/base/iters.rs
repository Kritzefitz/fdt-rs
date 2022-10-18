//! Iterative parsers of a [`DevTree`].
use core::mem::size_of;
use core::num::NonZeroUsize;
use core::str::from_utf8;

use crate::prelude::*;

use crate::base::parse::{next_devtree_token, ParsedTok};
use crate::base::{DevTree, DevTreeItem, DevTreeNode, DevTreeProp};
use crate::error::{DevTreeError, Result};
use crate::spec::fdt_reserve_entry;

// Re-export the basic parse iterator.
pub use super::parse::DevTreeParseIter;
pub use crate::common::prop::StringPropIter;

use fallible_iterator::FallibleIterator;

/// An iterator over [`fdt_reserve_entry`] objects within the FDT.
#[derive(Clone)]
pub struct DevTreeReserveEntryIter<'a, 'dt: 'a> {
    offset: usize,
    fdt: &'a DevTree<'dt>,
}

impl<'a, 'dt: 'a> DevTreeReserveEntryIter<'a, 'dt> {
    pub(crate) fn new(fdt: &'a DevTree<'dt>) -> Self {
        Self {
            offset: fdt.off_mem_rsvmap(),
            fdt,
        }
    }

    /// Return the current offset as a fdt_reserve_entry reference.
    ///
    /// # Safety
    ///
    /// The caller must verify that the current offset of this iterator is 32-bit aligned.
    /// (Each field is 32-bit aligned and they may be read individually.)
    unsafe fn read(&'a self) -> Result<&'dt fdt_reserve_entry> {
        Ok(&*self.fdt.ptr_at(self.offset)?)
    }
}

impl<'a, 'dt: 'a> Iterator for DevTreeReserveEntryIter<'a, 'dt> {
    type Item = &'dt fdt_reserve_entry;
    fn next(&mut self) -> Option<Self::Item> {
        if self.offset > self.fdt.totalsize() {
            None
        } else {
            // We guaruntee the read will be aligned to 32 bits because:
            // - We construct with guarunteed 32-bit aligned offset
            // - We always increment by an aligned amount
            let ret = unsafe { self.read().unwrap() };

            if ret.address == 0.into() && ret.size == 0.into() {
                return None;
            }
            self.offset += size_of::<fdt_reserve_entry>();
            Some(ret)
        }
    }
}

pub trait DevTreeIterator<'a, 'dt: 'a> {
    fn next_item_with_depth(&mut self) -> Result<Option<(DevTreeItem<'a, 'dt>, isize)>>;

    fn next_item(&mut self) -> Result<Option<DevTreeItem<'a, 'dt>>> {
        self.next_item_with_depth().map(|o| o.map(|(item, _)| item))
    }

    fn next_prop(&mut self) -> Result<Option<DevTreeProp<'a, 'dt>>> {
        loop {
            match self.next_item() {
                Ok(Some(DevTreeItem::Prop(p))) => return Ok(Some(p)),
                Ok(Some(_n)) => continue,
                Ok(None) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
    }

    fn next_node(&mut self) -> Result<Option<DevTreeNode<'a, 'dt>>> {
        loop {
            match self.next_item() {
                Ok(Some(DevTreeItem::Node(n))) => return Ok(Some(n)),
                Ok(Some(_p)) => continue,
                Ok(None) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
    }

    fn next_node_prop(&mut self) -> Result<Option<DevTreeProp<'a, 'dt>>> {
        match self.next_item() {
            // Return if a new node or an EOF.
            Ok(Some(item)) => Ok(item.prop()),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn next_compatible_node(&mut self, string: &str) -> Result<Option<DevTreeNode<'a, 'dt>>> {
        // If there is another node, advance our iterator to that node.
        self.next_node().and_then(|_| {
            // Iterate through all remaining properties in the tree looking for the compatible
            // string.
            loop {
                match self.next_prop() {
                    Ok(Some(prop)) => {
                        if prop.name()? == "compatible" && prop.str()? == string {
                            return Ok(Some(prop.node()));
                        }
                        continue;
                    }
                    Ok(None) => return Ok(None),
                    Err(e) => return Err(e),
                }
            }
        })
    }
}

/// An iterator over all [`DevTreeItem`] objects.
#[derive(Clone, PartialEq)]
pub struct DevTreeIter<'a, 'dt: 'a> {
    /// Offset of the last opened Device Tree Node.
    /// This is used to set properties' parent DevTreeNode.
    ///
    /// As defined by the spec, DevTreeProps must preceed Node definitions.
    /// Therefore, once a node has been closed this offset is reset to None to indicate no
    /// properties should follow.
    current_prop_parent_off: Option<NonZeroUsize>,

    /// Current offset into the flattened dt_struct section of the device tree.
    offset: usize,

    /// The depth we are currently parsing at. 0 is the level of the
    /// root node, -1 is the level of an imaginary parent of our root
    /// element and going one element down increases the depth by 1.
    depth: isize,
    pub(crate) fdt: &'a DevTree<'dt>,
}

#[derive(Clone, PartialEq)]
pub struct DevTreeNodeIter<'a, 'dt: 'a>(pub DevTreeIter<'a, 'dt>);
impl<'a, 'dt: 'a> FallibleIterator for DevTreeNodeIter<'a, 'dt> {
    type Item = DevTreeNode<'a, 'dt>;
    type Error = DevTreeError;
    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.0.next_node()
    }
}

#[derive(Clone, PartialEq)]
pub struct DevTreePropIter<'a, 'dt: 'a>(pub DevTreeIter<'a, 'dt>);
impl<'a, 'dt: 'a> FallibleIterator for DevTreePropIter<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeProp<'a, 'dt>;
    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.0.next_prop()
    }
}

#[derive(Clone, PartialEq)]
pub struct DevTreeNodePropIter<'a, 'dt: 'a>(pub DevTreeIter<'a, 'dt>);
impl<'a, 'dt: 'a> FallibleIterator for DevTreeNodePropIter<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeProp<'a, 'dt>;
    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.0.next_node_prop()
    }
}

#[derive(Clone, PartialEq)]
pub struct DevTreeCompatibleNodeIter<'s, 'a, 'dt: 'a> {
    pub iter: DevTreeIter<'a, 'dt>,
    pub string: &'s str,
}
impl<'s, 'a, 'dt: 'a> FallibleIterator for DevTreeCompatibleNodeIter<'s, 'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeNode<'a, 'dt>;
    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.iter.next_compatible_node(self.string)
    }
}

impl<'a, 'dt: 'a> DevTreeIter<'a, 'dt> {
    pub fn new(fdt: &'a DevTree<'dt>) -> Self {
        Self {
            offset: fdt.off_dt_struct(),
            current_prop_parent_off: None,
            // Initially we haven't parsed the root node, so if 0 is
            // supposed to be the root level, we are one level up from
            // that.
            depth: -1,
            fdt,
        }
    }

    fn current_node_itr(&self) -> Option<DevTreeIter<'a, 'dt>> {
        self.current_prop_parent_off.map(|offset| DevTreeIter {
            fdt: self.fdt,
            current_prop_parent_off: Some(offset),
            offset: offset.get(),
            depth: self.depth,
        })
    }

    pub fn last_node(mut self) -> Option<DevTreeNode<'a, 'dt>> {
        if let Some(off) = self.current_prop_parent_off.take() {
            self.offset = off.get();
            return self.next_node().unwrap();
        }
        None
    }
}

impl<'a, 'dt: 'a> DevTreeIterator<'a, 'dt> for DevTreeIter<'a, 'dt> {
    fn next_item_with_depth(&mut self) -> Result<Option<(DevTreeItem<'a, 'dt>, isize)>> {
        loop {
            let old_offset = self.offset;
            // Safe because we only pass offsets which are returned by next_devtree_token.
            let res = unsafe { next_devtree_token(self.fdt.buf(), &mut self.offset)? };

            match res {
                Some(ParsedTok::BeginNode(node)) => {
                    self.depth += 1;
                    self.current_prop_parent_off =
                        unsafe { Some(NonZeroUsize::new_unchecked(old_offset)) };
                    return Ok(Some((
                        DevTreeItem::Node(DevTreeNode {
                            parse_iter: self.clone(),
                            name: from_utf8(node.name).map_err(|e| e.into()),
                        }),
                        self.depth,
                    )));
                }
                Some(ParsedTok::Prop(prop)) => {
                    // Prop must come after a node.
                    let prev_node = match self.current_node_itr() {
                        Some(n) => n,
                        None => return Err(DevTreeError::ParseError),
                    };

                    return Ok(Some((
                        DevTreeItem::Prop(DevTreeProp::new(
                            prev_node,
                            prop.prop_buf,
                            prop.name_offset,
                        )),
                        self.depth,
                    )));
                }
                Some(ParsedTok::EndNode) => {
                    // The current node has ended.
                    // No properties may follow until the next node starts.
                    self.current_prop_parent_off = None;
                    self.depth -= 1;
                }
                Some(_) => continue,
                None => return Ok(None),
            }
        }
    }
}

impl<'a, 'dt: 'a> FallibleIterator for DevTreeIter<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeItem<'a, 'dt>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.next_item()
    }
}

#[derive(Clone, PartialEq)]
struct DevTreeMinDepthIter<'a, 'dt: 'a> {
    iter: DevTreeIter<'a, 'dt>,
    min_depth: isize,
    ended: bool,
}

impl<'a, 'dt: 'a> DevTreeMinDepthIter<'a, 'dt> {
    pub fn new(iter: DevTreeIter<'a, 'dt>, min_depth: isize) -> Self {
        Self {
            iter,
            min_depth,
            ended: false,
        }
    }
}

impl<'a, 'dt: 'dt> DevTreeIterator<'a, 'dt> for DevTreeMinDepthIter<'a, 'dt> {
    fn next_item_with_depth(&mut self) -> Result<Option<(DevTreeItem<'a, 'dt>, isize)>> {
        if self.ended {
            return Ok(None);
        }
        loop {
            match self.iter.next_item_with_depth() {
                Ok(Some((item, depth))) => {
                    if depth >= self.min_depth {
                        break Ok(Some((item, depth)));
                    } else {
                        if let DevTreeItem::Node(_) = item {
                            self.ended = true;
                            break Ok(None);
                        }
                    }
                }
                r => break r,
            }
        }
    }
}

impl<'a, 'dt> FallibleIterator for DevTreeMinDepthIter<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeItem<'a, 'dt>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.next_item()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, max) = self.iter.size_hint();
        (0, max)
    }
}

#[derive(Clone, PartialEq)]
struct DevTreeDepthIter<'a, 'dt: 'a> {
    iter: DevTreeIter<'a, 'dt>,
    depth: isize,
    ended: bool,
}

impl<'a, 'dt: 'a> DevTreeDepthIter<'a, 'dt> {
    fn new(iter: DevTreeIter<'a, 'dt>, depth: isize) -> Self {
        Self {
            iter,
            depth,
            ended: false,
        }
    }
}

impl<'a, 'dt> DevTreeIterator<'a, 'dt> for DevTreeDepthIter<'a, 'dt> {
    fn next_item_with_depth(&mut self) -> Result<Option<(DevTreeItem<'a, 'dt>, isize)>> {
        if self.ended {
            return Ok(None);
        }
        loop {
            match self.iter.next_item_with_depth() {
                Ok(Some((item, depth))) => {
                    if depth == self.depth {
                        break Ok(Some((item, depth)));
                    } else if depth < self.depth {
                        self.ended = true;
                        break Ok(None);
                    }
                }
                r => break r,
            }
        }
    }
}

impl<'a, 'dt> FallibleIterator for DevTreeDepthIter<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeItem<'a, 'dt>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.next_item()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, max) = self.iter.size_hint();
        (0, max)
    }
}

#[derive(Clone, PartialEq)]
struct DevTreeSkipCurrentIter<I> {
    iter: I,
    target_depth: isize,
    done: bool,
}

impl<I> DevTreeSkipCurrentIter<I> {
    fn new(iter: I, current_depth: isize) -> Self {
        Self {
            iter,
            target_depth: current_depth,
            done: false,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    where
        I: FallibleIterator,
    {
        let (_, max) = self.iter.size_hint();
        (0, max)
    }
}

impl<'a, 'dt: 'a, I> DevTreeIterator<'a, 'dt> for DevTreeSkipCurrentIter<I>
where
    I: DevTreeIterator<'a, 'dt>,
{
    fn next_item_with_depth(&mut self) -> Result<Option<(DevTreeItem<'a, 'dt>, isize)>> {
        if self.done {
            self.iter.next_item_with_depth()
        } else {
            loop {
                if let Ok(Some((DevTreeItem::Node(node), depth))) = self.iter.next_item_with_depth()
                {
                    if depth == self.target_depth {
                        self.done = true;
                        break Ok(Some((DevTreeItem::Node(node), depth)));
                    }
                }
            }
        }
    }
}

/// A variant of [`DevTreeIter`] limited to only descendants of a
/// given node.
#[derive(Clone, PartialEq)]
pub struct DevTreeDescendantIter<'a, 'dt>(DevTreeMinDepthIter<'a, 'dt>);

impl<'a, 'dt> DevTreeDescendantIter<'a, 'dt> {
    pub(crate) fn new(iter: DevTreeIter<'a, 'dt>) -> Self {
        let target_depth = iter.depth + 1;
        Self(DevTreeMinDepthIter::new(iter, target_depth))
    }
}

impl<'a, 'dt> DevTreeIterator<'a, 'dt> for DevTreeDescendantIter<'a, 'dt> {
    fn next_item_with_depth(&mut self) -> Result<Option<(DevTreeItem<'a, 'dt>, isize)>> {
        self.0.next_item_with_depth()
    }
}

impl<'a, 'dt> FallibleIterator for DevTreeDescendantIter<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeItem<'a, 'dt>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.0.next()
    }
}

/// A variant of [`DevTreeIter`] limited to only direct children of a
/// given node.
#[derive(Clone, PartialEq)]
pub struct DevTreeChildrenIter<'a, 'dt>(DevTreeDepthIter<'a, 'dt>);

impl<'a, 'dt> DevTreeChildrenIter<'a, 'dt> {
    pub(crate) fn new(iter: DevTreeIter<'a, 'dt>) -> Self {
        let target_depth = iter.depth + 1;
        Self(DevTreeDepthIter::new(iter, target_depth))
    }
}

impl<'a, 'dt> DevTreeIterator<'a, 'dt> for DevTreeChildrenIter<'a, 'dt> {
    fn next_item_with_depth(&mut self) -> Result<Option<(DevTreeItem<'a, 'dt>, isize)>> {
        self.0.next_item_with_depth()
    }
}

impl<'a, 'dt> FallibleIterator for DevTreeChildrenIter<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeItem<'a, 'dt>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

/// A variant of [`DevTreeIter`] limited to only later siblings and
/// their descendants of a given node. Note that this will only yield
/// siblings that come after the reference node, but not any before
/// that node.
#[derive(Clone, PartialEq)]
pub struct DevTreeSiblingAndDescendantIterator<'a, 'dt>(
    DevTreeSkipCurrentIter<DevTreeMinDepthIter<'a, 'dt>>,
);

impl<'a, 'dt> DevTreeSiblingAndDescendantIterator<'a, 'dt> {
    pub(crate) fn new(iter: DevTreeIter<'a, 'dt>) -> Self {
        let current_depth = iter.depth;
        Self(DevTreeSkipCurrentIter::new(
            DevTreeMinDepthIter::new(iter, current_depth),
            current_depth,
        ))
    }
}

impl<'a, 'dt> DevTreeIterator<'a, 'dt> for DevTreeSiblingAndDescendantIterator<'a, 'dt> {
    fn next_item_with_depth(&mut self) -> Result<Option<(DevTreeItem<'a, 'dt>, isize)>> {
        self.0.next_item_with_depth()
    }
}

impl<'a, 'dt> FallibleIterator for DevTreeSiblingAndDescendantIterator<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeItem<'a, 'dt>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.0.next_item()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

/// A variant of [`DevTreeIter`] limited to only later siblings of a
/// given node. Note that this will only yield siblings that come
/// after the reference node, but not any before that node.
#[derive(Clone, PartialEq)]
struct DevTreeSiblingIter<'a, 'dt>(DevTreeSkipCurrentIter<DevTreeDepthIter<'a, 'dt>>);

impl<'a, 'dt> DevTreeSiblingIter<'a, 'dt> {
    pub(crate) fn new(iter: DevTreeIter<'a, 'dt>) -> Self {
        let current_depth = iter.depth;
        Self(DevTreeSkipCurrentIter::new(
            DevTreeDepthIter::new(iter, current_depth),
            current_depth,
        ))
    }
}

impl<'a, 'dt> DevTreeIterator<'a, 'dt> for DevTreeSiblingIter<'a, 'dt> {
    fn next_item_with_depth(&mut self) -> Result<Option<(DevTreeItem<'a, 'dt>, isize)>> {
        self.0.next_item_with_depth()
    }
}

impl<'a, 'dt> FallibleIterator for DevTreeSiblingIter<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeItem<'a, 'dt>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.0.next_item()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
