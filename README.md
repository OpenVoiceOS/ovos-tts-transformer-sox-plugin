# Sox TTS Transformer

This plugin changes TTS audio before playback. It runs after the TTS engine generates a wav file and applies [SoX](http://sox.sourceforge.net/) (Sound eXchange) effects to it. Use it to add effects such as reverb, echo, or pitch shifting to synthesized speech.

## Effects

The plugin supports these effects:

- Pitch shifting
- Phaser
- Flanger
- Reverb
- Tempo adjustment
- Treble adjustment
- Tremolo
- Reverse playback
- Speed adjustment
- Chorus
- Echo
- Bend
- Stretch
- Overdrive
- Bass adjustment
- Allpass
- Bandpass
- Bandreject
- Compand
- Contrast adjustment
- Equalizer
- Gain adjustment
- Highpass filter
- Lowpass filter
- Loudness
- Noise reduction

For the parameters of each effect, read the docstrings in the code and the [SoX documentation](http://sox.sourceforge.net/sox.html).

## Requirements

- Python 3.6 or later
- SoX

Install SoX and make sure it is on your system's PATH. The plugin calls the `sox` executable directly.

## Install

```bash
pip install ovos-tts-transformer-sox-plugin
```

## Usage

[OVOS](https://openvoiceos.org) loads this plugin as a TTS transformer through the `opm.transformer.tts` entry point. Configure the effects to apply in your TTS configuration:

```json
{
  "tts": {
    "transformers": {
      "ovos-tts-transformer-sox": {
        "default_effects": {
          "reverb": {},
          "pitch": {"n_semitones": 2}
        }
      }
    }
  }
}
```

Each key under `default_effects` names an effect, and its value holds the keyword arguments for that effect.

## Related projects

- [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager) — defines the `TTSTransformer` base class and the transformer plugin API.
- [OpenVoiceOS](https://github.com/OpenVoiceOS) — the org that maintains this plugin and the wider OVOS platform.

## License

Apache-2.0
