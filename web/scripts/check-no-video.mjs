// The public site carries no clip (PLAN.md D21): fail when a folder holds a video file or a moving image.
//
// A file is a hit when its extension is a video extension, or when its first bytes are a video container
// (so a renamed clip is still found), or when it is an animated GIF, WebP, PNG or AVIF (a clip in disguise).
//
//   node scripts/check-no-video.mjs [<dir>]     the default is web/out
import { closeSync, existsSync, openSync, readdirSync, readFileSync, readSync, statSync } from "node:fs";
import path from "node:path";
import { OUT_DIR, shown } from "./lib/paths.mjs";

const VIDEO_EXTENSIONS = new Set([
  "mp4", "m4v", "mov", "webm", "mkv", "avi", "wmv", "flv", "ogv", "mpg", "mpeg", "3gp", "m2ts", "mts", "ts",
  "m3u8", "mpd",
]);
const STILL_BRANDS = new Set(["avif", "heic", "heix", "mif1", "msf1"]);
const HEAD_BYTES = 1 << 16;

function head(file) {
  const buffer = Buffer.alloc(HEAD_BYTES);
  const handle = openSync(file, "r");
  try {
    return buffer.subarray(0, readSync(handle, buffer, 0, HEAD_BYTES, 0));
  } finally {
    closeSync(handle);
  }
}

function ascii(buffer, start, end) {
  return buffer.subarray(start, end).toString("latin1");
}

/** Why the bytes are a video or a moving image, or null. */
export function movingKind(bytes) {
  if (bytes.length >= 12 && ascii(bytes, 4, 8) === "ftyp") {
    const brand = ascii(bytes, 8, 12).trim().toLowerCase();
    if (brand === "avis") return "an animated AVIF";
    return STILL_BRANDS.has(brand) ? null : `an MP4 or QuickTime file (brand ${brand})`;
  }
  if (bytes.length >= 4 && bytes.readUInt32BE(0) === 0x1a45dfa3) return "a WebM or Matroska file";
  if (bytes.length >= 12 && ascii(bytes, 0, 4) === "RIFF" && ascii(bytes, 8, 12) === "AVI ") return "an AVI file";
  if (bytes.length >= 4 && ascii(bytes, 0, 4) === "OggS") return "an Ogg media file";
  if (bytes.length >= 3 && ascii(bytes, 0, 3) === "FLV") return "a Flash video file";
  if (bytes.length >= 16 && bytes.readUInt32BE(0) === 0x3026b275) return "a Windows Media file";
  if (bytes.length >= 12 && ascii(bytes, 0, 4) === "RIFF" && ascii(bytes, 8, 12) === "WEBP") {
    return bytes.includes(Buffer.from("ANIM")) ? "an animated WebP" : null;
  }
  if (bytes.length >= 8 && bytes.readUInt32BE(0) === 0x89504e47) {
    const idat = bytes.indexOf(Buffer.from("IDAT"));
    const actl = bytes.indexOf(Buffer.from("acTL"));
    return actl >= 0 && (idat < 0 || actl < idat) ? "an animated PNG" : null;
  }
  if (bytes.length >= 6 && /^GIF8[79]a$/.test(ascii(bytes, 0, 6))) {
    // More than one image descriptor after a graphic control extension means more than one frame.
    let frames = 0;
    for (let index = 13; index < bytes.length - 8; index += 1) {
      if (bytes[index] === 0x21 && bytes[index + 1] === 0xf9 && bytes[index + 2] === 0x04 && bytes[index + 7] === 0x00) {
        frames += 1;
        if (frames > 1) return "an animated GIF";
      }
    }
    return null;
  }
  return null;
}

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else if (entry.isFile()) out.push(full);
  }
  return out;
}

/** Every video or moving image under `dir`, as { file, why }. */
export function findVideo(dir) {
  const hits = [];
  for (const file of walk(dir).sort()) {
    const extension = path.extname(file).slice(1).toLowerCase();
    if (VIDEO_EXTENSIONS.has(extension)) {
      hits.push({ file, why: `a .${extension} file` });
      continue;
    }
    let bytes = head(file);
    // The frames of a GIF can sit anywhere in the file, so a GIF is read whole.
    if (bytes.length >= 6 && /^GIF8[79]a$/.test(ascii(bytes, 0, 6))) bytes = readFileSync(file);
    const why = movingKind(bytes);
    if (why) hits.push({ file, why });
  }
  return hits;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const dir = process.argv[2] ? path.resolve(process.argv[2]) : OUT_DIR;
  if (!existsSync(dir) || !statSync(dir).isDirectory()) {
    console.error(`video: no folder at ${shown(dir)}`);
    process.exitCode = 1;
  } else {
    const hits = findVideo(dir);
    for (const hit of hits) console.error(`video: ${path.relative(dir, hit.file)} is ${hit.why}`);
    if (hits.length) {
      console.error(`video: ${hits.length} video file(s) or moving image(s) in ${shown(dir)}`);
      process.exitCode = 1;
    } else {
      console.log(`video: no video file and no moving image in ${shown(dir)}`);
    }
  }
}
