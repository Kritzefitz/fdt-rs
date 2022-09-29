#[cfg(doc)]
use super::*;

use crate::base::iters::{DevTreeIter, DevTreeNodeIter, DevTreeNodePropIter};
use crate::error::Result;
use crate::prelude::FallibleIterator;

/// A handle to a Device Tree Node within the device tree.
#[derive(Clone)]
pub struct DevTreeNode<'a, 'dt: 'a> {
    pub(super) name: Result<&'dt str>,
    pub(super) parse_iter: DevTreeIter<'a, 'dt>,
}

impl<'a, 'dt: 'a> PartialEq for DevTreeNode<'a, 'dt> {
    fn eq(&self, other: &Self) -> bool {
        self.parse_iter == other.parse_iter
    }
}

impl<'a, 'dt: 'a> DevTreeNode<'a, 'dt> {
    /// Returns the name of the `DevTreeNode` (including unit address tag)
    #[inline]
    pub fn name(&'a self) -> Result<&'dt str> {
        self.name
    }

    /// Returns an iterator over this node's children [`DevTreeProp`]
    #[must_use]
    pub fn props(&'a self) -> DevTreeNodePropIter<'a, 'dt> {
        DevTreeNodePropIter(self.parse_iter.clone())
    }

    /// Returns the next [`DevTreeNode`] object with the provided compatible device tree property
    /// or `None` if none exists.
    ///
    /// # Example
    ///
    /// The following example iterates through all nodes with compatible value "virtio,mmio"
    /// and prints each node's name.
    ///
    /// TODO
    pub fn find_next_compatible_node(&self, string: &str) -> Result<Option<DevTreeNode<'a, 'dt>>> {
        self.parse_iter.clone().next_compatible_node(string)
    }

    /// Return an iterator over all descendants of this node.
    pub fn descendants(&self) -> DevTreeIter<'a, 'dt> {
        let mut i = self.parse_iter.clone();
        i.limit_to_descendants();
        i
    }

    /// Return an iterator over all descendant nodes of this node.
    pub fn descendant_nodes(&self) -> DevTreeNodeIter<'a, 'dt> {
        DevTreeNodeIter(self.descendants())
    }

    /// Return and iterator over all direct children of this node.
    pub fn children(&self) -> DevTreeIter<'a, 'dt> {
        let mut i = self.parse_iter.clone();
        i.limit_to_children();
        i
    }

    /// Return an interator over all direct children nodes of this node.
    pub fn child_nodes(&self) -> DevTreeNodeIter<'a, 'dt> {
        DevTreeNodeIter(self.children())
    }

    /// Return an iterator over all descendants of this node, all
    /// siblings after the node and their descendants. If you only
    /// want siblings and their descendants, but not this node's
    /// descendants, first use [`next_sibling`](Self::next_sibling) and then
    /// [`siblings_and_descendants`](Self::siblings_and_descendants) on that.
    pub fn siblings_and_descendants(&self) -> DevTreeIter<'a, 'dt> {
        let mut i = self.parse_iter.clone();
        i.limit_to_siblings_and_descendants();
        i
    }

    /// Return and iterator over all descendant nodes and following
    /// siblings nodes and their descendant nodes.
    pub fn sibling_and_descendant_nodes(&self) -> DevTreeNodeIter<'a, 'dt> {
        DevTreeNodeIter(self.siblings_and_descendants())
    }

    /// Return an iterator over all later siblings of this node. Note
    /// that this is dependant on the order of the siblings in the
    /// DevTree as this can only return siblings that come after this
    /// node, not previous siblings.
    pub fn siblings(&self) -> DevTreeIter<'a, 'dt> {
        let mut i = self.parse_iter.clone();
        i.limit_to_siblings();
        i
    }

    /// Return and iterator over all later sibling nodes of this node.
    pub fn sibling_nodes(&self) -> DevTreeNodeIter<'a, 'dt> {
        DevTreeNodeIter(self.siblings_and_descendants())
    }

    /// Return the next sibling of this node.
    pub fn next_sibling(&self) -> Result<Option<DevTreeNode<'a, 'dt>>> {
        self.siblings().next_node()
    }

    /// Returns the [`DevTreeNode`] at the given path below this node.
    pub fn node_at_path<'s, I>(&self, path: I) -> Result<Option<DevTreeNode<'a, 'dt>>>
    where
        I: Iterator<Item = &'s str>,
    {
        let mut current = self.clone();
        for component in path {
            if let Some(found) = current
                .child_nodes()
                .find(|node| Ok(node.name()? == component))?
            {
                current = found;
            } else {
                return Ok(None);
            }
        }
        Ok(Some(current))
    }
}
