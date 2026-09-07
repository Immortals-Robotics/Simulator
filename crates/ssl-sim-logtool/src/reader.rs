//! Streaming reader for SSL log files (`SSL_LOG_FILE` v1, optionally gzip).
//!
//! Record layout (big-endian): `i64` receive timestamp [ns], `i32` type,
//! `i32` size, payload. Types: 2 legacy vision, 3 referee, 4 vision
//! (`SSL_WrapperPacket`), 5 tracker (`TrackerWrapperPacket`), 6 index.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};
use flate2::read::MultiGzDecoder;
use prost::Message;
use ssl_sim_proto::{gc, sim, tracked};

/// One decoded record.
#[derive(Debug, Clone)]
pub enum Record {
    /// SSL vision wrapper packet (detection and/or geometry).
    Vision(sim::SslWrapperPacket),
    /// Game-controller referee message.
    Referee(gc::Referee),
    /// Tracker (autoref / GC ball tracker) packet.
    Tracker(tracked::TrackerWrapperPacket),
    /// Anything else (legacy vision, index, unknown).
    #[allow(dead_code)]
    Other {
        /// Raw type id.
        kind: i32,
        /// Payload length.
        len: usize,
    },
}

/// A record with its log receive time.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Receive timestamp [ns] as written by the logger (wall clock).
    pub time_ns: i64,
    /// Decoded record.
    pub record: Record,
}

/// Streaming iterator over a log file.
pub struct LogReader {
    input: Box<dyn Read>,
    buf: Vec<u8>,
    /// Records read so far.
    pub count: u64,
    /// Records that failed to decode, by type.
    pub decode_errors: u64,
}

impl LogReader {
    /// Open a `.log` or `.log.gz` file and validate the header.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let reader = BufReader::with_capacity(1 << 20, file);
        let mut input: Box<dyn Read> = if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gz"))
        {
            Box::new(MultiGzDecoder::new(reader))
        } else {
            Box::new(reader)
        };
        let mut header = [0u8; 16];
        input.read_exact(&mut header).context("read log header")?;
        if &header[..12] != b"SSL_LOG_FILE" {
            bail!(
                "not an SSL log file: header {:?}",
                String::from_utf8_lossy(&header[..12])
            );
        }
        let version = i32::from_be_bytes(header[12..16].try_into().unwrap());
        if version != 1 {
            bail!("unsupported SSL log version {version}");
        }
        Ok(Self {
            input,
            buf: Vec::with_capacity(1 << 16),
            count: 0,
            decode_errors: 0,
        })
    }

    /// Read the next record; `Ok(None)` at end of file.
    pub fn next_entry(&mut self) -> Result<Option<Entry>> {
        let mut meta = [0u8; 16];
        match self.input.read_exact(&mut meta) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let time_ns = i64::from_be_bytes(meta[0..8].try_into().unwrap());
        let kind = i32::from_be_bytes(meta[8..12].try_into().unwrap());
        let size = i32::from_be_bytes(meta[12..16].try_into().unwrap());
        if size < 0 {
            bail!("negative record size {size} at record {}", self.count);
        }
        self.buf.clear();
        self.buf.resize(size as usize, 0);
        self.input
            .read_exact(&mut self.buf)
            .context("read record payload")?;
        self.count += 1;
        let record = match kind {
            4 => match sim::SslWrapperPacket::decode(self.buf.as_slice()) {
                Ok(p) => Record::Vision(p),
                Err(_) => {
                    self.decode_errors += 1;
                    Record::Other {
                        kind,
                        len: self.buf.len(),
                    }
                }
            },
            3 => match gc::Referee::decode(self.buf.as_slice()) {
                Ok(r) => Record::Referee(r),
                Err(_) => {
                    self.decode_errors += 1;
                    Record::Other {
                        kind,
                        len: self.buf.len(),
                    }
                }
            },
            5 => match tracked::TrackerWrapperPacket::decode(self.buf.as_slice()) {
                Ok(t) => Record::Tracker(t),
                Err(_) => {
                    self.decode_errors += 1;
                    Record::Other {
                        kind,
                        len: self.buf.len(),
                    }
                }
            },
            _ => Record::Other {
                kind,
                len: self.buf.len(),
            },
        };
        Ok(Some(Entry { time_ns, record }))
    }
}

impl Iterator for LogReader {
    type Item = Result<Entry>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_entry().transpose()
    }
}
