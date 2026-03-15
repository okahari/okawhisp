# Sound Files

Optional sound files for recording start/stop feedback.
Played via `paplay` (PulseAudio/PipeWire) for correct audio routing.

- **start.wav** / **start.mp3**: Recording start sound
- **stop.wav** / **stop.mp3**: Recording stop sound

WAV files are preferred (played directly by paplay). MP3 files are auto-converted to WAV via ffmpeg on first use.

If no custom sound files are found, the script generates synthetic sounds (switch click for start, soft buzzer for stop).
