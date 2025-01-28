//! Velo: VErifying maximum link LOads in a changing world

#![deny(missing_docs, missing_debug_implementations)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_lifetimes)]

use analysis::Velo;
use indicatif::{ProgressBar, ProgressBarIter, ProgressFinish, ProgressIterator, ProgressStyle};

pub mod algorithms;
pub mod analysis;
pub mod explorer;
pub mod performance;
pub mod scenario;
#[cfg(test)]
mod tests;
pub mod traffic_matrix;
pub mod utils;

pub(crate) const PROGRESS_TEMPLATE: &str =
    "{msg:50} {pos:>9}/{len:<9} {elapsed:>3}/{eta:<3} [{wide_bar}] {percent:>3}%";

pub(crate) const SPINNER_TEMPLATE: &str = "{msg:81} {elapsed:<3}     {spinner} ({pos} iter)";

/// The shared progress style
pub fn progress_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(PROGRESS_TEMPLATE)
        .unwrap()
        .progress_chars("##-")
}

pub(crate) fn spinner_style() -> ProgressStyle {
    ProgressStyle::default_spinner().template(SPINNER_TEMPLATE).unwrap()
}

pub(crate) fn my_progress(
    msg: impl Into<String>,
    len: usize,
    keep: bool,
    show: bool,
) -> ProgressBar {
    if show {
        ProgressBar::new(len as u64)
            .with_style(progress_style())
            .with_finish(if keep {
                ProgressFinish::AndLeave
            } else {
                ProgressFinish::AndClear
            })
            .with_message(msg.into())
    } else {
        ProgressBar::hidden()
    }
}

pub(crate) fn my_spinner(msg: impl Into<String>, keep: bool, show: bool) -> ProgressBar {
    if show {
        ProgressBar::new_spinner()
            .with_style(spinner_style())
            .with_finish(if keep {
                ProgressFinish::AndLeave
            } else {
                ProgressFinish::AndClear
            })
            .with_message(msg.into())
    } else {
        ProgressBar::hidden()
    }
}

pub(crate) trait MyProgressIterator
where
    Self: Sized + Iterator,
{
    /// Wrap an iterator with a custom progress bar.
    fn my_progress_with(self, progress: ProgressBar) -> ProgressBarIter<Self>;

    /// Wrap an iterator with the default spinner.
    fn my_spinner(self, msg: impl Into<String>, keep: bool, show: bool) -> ProgressBarIter<Self> {
        self.my_progress_with(my_spinner(msg, keep, show))
    }

    /// Wrap an iterator with default styling.
    fn my_progress_config<P>(
        self,
        msg: impl Into<String>,
        keep: bool,
        config: &Velo<P>,
    ) -> ProgressBarIter<Self>
    where
        Self: ExactSizeIterator,
    {
        let len = self.len();
        self.my_progress_count_config(msg, len, keep, config)
    }

    /// Wrap an iterator with default styling.
    fn my_progress(self, msg: impl Into<String>, keep: bool, show: bool) -> ProgressBarIter<Self>
    where
        Self: ExactSizeIterator,
    {
        let len = self.len();
        self.my_progress_count(msg, len, keep, show)
    }

    /// Wrap an iterator with an explicit element count and default styling.
    fn my_progress_count_config<P>(
        self,
        msg: impl Into<String>,
        len: usize,
        keep: bool,
        config: &Velo<P>,
    ) -> ProgressBarIter<Self> {
        self.my_progress_with(config.progress_bar(msg, len, keep))
    }

    /// Wrap an iterator with an explicit element count and default styling.
    fn my_progress_count(
        self,
        msg: impl Into<String>,
        len: usize,
        keep: bool,
        show: bool,
    ) -> ProgressBarIter<Self> {
        self.my_progress_with(my_progress(msg, len, keep, show))
    }
}

impl<I, T: Iterator<Item = I>> MyProgressIterator for T {
    fn my_progress_with(self, progress: ProgressBar) -> ProgressBarIter<Self> {
        self.progress_with(progress)
    }
}
