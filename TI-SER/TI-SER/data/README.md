# Data layout

Audio is **not** included. Place the corpora here (WAV only):

```
data/
  RAVDESS/clean/*.wav     # may be nested, e.g. Actor_01/*.wav (speaker id is read from the Actor_XX folder)
  TESS/clean/*.wav        # filenames like OAF_back_angry.wav / YAF_door_pleasant_surprise.wav
  CREMA-D/clean/*.wav     # optional; only needed for the CREMA-D block
```

* RAVDESS: only speech files (vocal channel `01`) are used; *calm* is merged into *neutral*.
* TESS: *pleasant surprise* is mapped to *surprise*.
* If RAVDESS/TESS are missing and the `kaggle` CLI is configured, the code will try to download them
  (`uwrfkaggler/ravdess-emotional-speech-audio`, `ejlok1/toronto-emotional-speech-set-tess`).
* Dataset licences differ (TESS is CC BY-NC 4.0); check each before redistributing.
