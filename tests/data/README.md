# Test data

Media files here are git-ignored (see `.gitignore`); this file records where they came from
so anyone can fetch them again. All are from archive.org.

| File | Source item | Length | Rights as stated on archive.org |
|---|---|---|---|
| `jfk.wav` | whisper.cpp `samples/jfk.wav` (JFK inaugural address excerpt) | 11 s | US government work |
| `public/Doctorin1946_512kb.mp4` | `Doctorin1946` (Prelinger Archives) | 18:47 | public domain (licenseurl: creativecommons.org/licenses/publicdomain) |
| `public/FromtheG1954_512kb.mp4` | `FromtheG1954` (Prelinger Archives) | 19:04 | public domain |
| `public/marktwainsspeeches_002_twain_64kb.mp3` | `marktwains_speeches_1309_librivox` track 002 | 25:42 | public domain mark (LibriVox) |
| `public/marktwainsspeeches_060_twain_64kb.mp3` | same item, track 060 | 24:00 | public domain mark (LibriVox) |

Fetch with e.g.
`curl -LO https://archive.org/download/Doctorin1946/Doctorin1946_512kb.mp4`

Not used on purpose: `SanFrancisco1955CinemascopeFilm` (CC BY-SA, not public domain).
`jfk_video.mp4` is `jfk.wav` muxed over a test pattern with ffmpeg.
