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

    /// The depth we are currently parsing at, relative to the root
    /// this iterator has been localized to or the root of the entire
    /// tree by default. 0 is the level of our root nodes, -1 is the
    /// level of the parent of our root elements and going one element
    /// down increases the depth by 1.
    depth: isize,

    /// Flag indicating that the iterator should only return items at
    /// the root it has been localized to. Useful for parsing only the
    /// children or siblings of a node.
    shallow: bool,

    /// Flag indicating that all properties of the current node
    /// should be discarded. Useful in certain situations when the
    /// scope of the iterator is changed to exclude the current node.
    skip_current_node_props: bool,

    /// Flag indicating that the iterator has reached the end of its
    /// local scope and should not continue parsing.
    local_end_reached: bool,
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
            shallow: false,
            skip_current_node_props: false,
            local_end_reached: false,
            fdt,
        }
    }

    fn current_node_itr(&self) -> Option<DevTreeIter<'a, 'dt>> {
        if self.skip_current_node_props {
            // In this case the current node isn't actually part of
            // our scope anymore and we should act as if it doesn't
            // exist.
            None
        } else {
            self.current_prop_parent_off.map(|offset| DevTreeIter {
                fdt: self.fdt,
                current_prop_parent_off: Some(offset),
                offset: offset.get(),
                depth: self.depth,
                shallow: self.shallow,
                skip_current_node_props: false,
                local_end_reached: false,
            })
        }
    }

    pub fn last_node(mut self) -> Option<DevTreeNode<'a, 'dt>> {
        if self.skip_current_node_props {
            // In this case the current node isn't actually part of
            // our scope anymore and we should act as if it doesn't
            // exist.
            None
        } else if let Some(off) = self.current_prop_parent_off.take() {
            self.offset = off.get();
            return self.next_node().unwrap();
        } else {
            None
        }
    }

    pub fn next_item(&mut self) -> Result<Option<DevTreeItem<'a, 'dt>>> {
        if self.local_end_reached {
            return Ok(None);
        }
        loop {
            let old_offset = self.offset;
            // Safe because we only pass offsets which are returned by next_devtree_token.
            let res = unsafe { next_devtree_token(self.fdt.buf(), &mut self.offset)? };

            match res {
                Some(ParsedTok::BeginNode(node)) => {
                    self.depth += 1;
                    self.skip_current_node_props = false;
                    self.current_prop_parent_off =
                        unsafe { Some(NonZeroUsize::new_unchecked(old_offset)) };
                    if self.depth < 0 || self.shallow && self.depth != 0 {
                        continue;
                    }
                    return Ok(Some(DevTreeItem::Node(DevTreeNode {
                        parse_iter: self.clone(),
                        name: from_utf8(node.name).map_err(|e| e.into()),
                    })));
                }
                Some(ParsedTok::Prop(prop)) => {
                    if self.depth < 0
                        || self.shallow && self.depth != 0
                        || self.skip_current_node_props
                    {
                        continue;
                    }
                    // Prop must come after a node.
                    let prev_node = match self.current_node_itr() {
                        Some(n) => n,
                        None => return Err(DevTreeError::ParseError),
                    };

                    return Ok(Some(DevTreeItem::Prop(DevTreeProp::new(
                        prev_node,
                        prop.prop_buf,
                        prop.name_offset,
                    ))));
                }
                Some(ParsedTok::EndNode) => {
                    // The current node has ended.
                    // No properties may follow until the next node starts.
                    self.current_prop_parent_off = None;
                    if self.depth < 0 {
                        // We were already in the parent of our roots,
                        // going up would leave our scope, so we end
                        // here.
                        self.local_end_reached = true;
                        return Ok(None);
                    }
                    self.depth -= 1;
                }
                Some(_) => continue,
                None => return Ok(None),
            }
        }
    }

    pub fn next_prop(&mut self) -> Result<Option<DevTreeProp<'a, 'dt>>> {
        loop {
            match self.next() {
                Ok(Some(DevTreeItem::Prop(p))) => return Ok(Some(p)),
                Ok(Some(_n)) => continue,
                Ok(None) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
    }

    pub fn next_node(&mut self) -> Result<Option<DevTreeNode<'a, 'dt>>> {
        loop {
            match self.next() {
                Ok(Some(DevTreeItem::Node(n))) => return Ok(Some(n)),
                Ok(Some(_p)) => continue,
                Ok(None) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
    }

    pub fn next_node_prop(&mut self) -> Result<Option<DevTreeProp<'a, 'dt>>> {
        match self.next() {
            // Return if a new node or an EOF.
            Ok(Some(item)) => Ok(item.prop()),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn next_compatible_node(&mut self, string: &str) -> Result<Option<DevTreeNode<'a, 'dt>>> {
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

    /// Limit the scope of the iterator to the descendants of the last
    /// node. If no previous node exists, this will set the scope to
    /// the entire tree. If the previous node has been excluded from
    /// the current scope, this will still limit the scope to
    /// descendants of that node.
    pub(crate) fn limit_to_descendants(&mut self) {
        assert!(
            self.depth >= -1,
            "Scope is already more limited than expected at {}",
            self.depth
        );
        self.depth = -1;
        self.shallow = false;
        self.skip_current_node_props = false;
    }

    /// Limit the scope of the iterator to direct children of the last
    /// node. If no previous node exists, this will limit the scope to
    /// only the root element. If the previous node has been excluded
    /// from the scope, this will still limit the scope to that nodes
    /// children.
    pub(crate) fn limit_to_children(&mut self) {
        assert!(
            self.depth >= -1,
            "Scope is already more limited than expected at {}",
            self.depth
        );
        self.depth = -1;
        self.shallow = true;
        self.skip_current_node_props = false;
    }

    /// Limit the scope the iterator to descendants of the last node,
    /// its descendants, its later siblings and all their descendants,
    /// but not any props of the last returned node that might follow.
    ///
    /// Note that only siblings and descendants thereof will be
    /// included that come after the last node. The iterator won't go
    /// back to any nodes that come before the last node.
    ///
    /// Also note that the new scope will include the descendants of
    /// the last node. If you want only siblings and their
    /// descendants, without the last nodes descendants, first call
    /// [`limit_to_siblings`](Self::limit_to_siblings), then advance to the first siblings with
    /// [`next_node`](Self::next_node) and then set the final desired scope with
    /// [`limit_to_siblings_and_descendants`](Self::limit_to_siblings_and_descendants)
    ///
    /// If there is no previous node, this will set the scope to the
    /// entire tree. If the previous node has been excluded from the
    /// current scope, this will widen the scope just as if that node
    /// was still in scope.
    pub(crate) fn limit_to_siblings_and_descendants(&mut self) {
        // This introduces an interesting edge case. If this is called
        // before any node is parsed, the depth will increase from -1
        // to 0, as if there was some root element above the actual
        // root. But this shouldn't be a problem, since that imaginary
        // root-above-the-root doesn't actually have any sibilings we
        // might accidentally include this way.
        self.depth = 0;
        self.shallow = false;
        self.skip_current_node_props = true;
    }

    /// Limit the scope of the iterator to only direct sibilings of
    /// the last node that come after that node. The parser won't
    /// return to earlier siblings. If there is no previous node
    /// (because you didn't parse the root node yet), this will
    /// effectively limit the scope to nothing, because we assume the
    /// existence of an implicit super-root above the actual root that
    /// doesn't have any siblings. If the last node is excluded from
    /// the current scope, this will still set the scope relative to
    /// that node as if it was still in scope.
    pub(crate) fn limit_to_siblings(&mut self) {
        // This introduces the same edge case as
        // limit_to_siblings_and_descendants.
        self.depth = 0;
        self.shallow = true;
        self.skip_current_node_props = true;
    }
}

impl<'a, 'dt: 'a> FallibleIterator for DevTreeIter<'a, 'dt> {
    type Error = DevTreeError;
    type Item = DevTreeItem<'a, 'dt>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        self.next_item()
    }
}
