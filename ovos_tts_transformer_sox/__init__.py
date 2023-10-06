from distutils.spawn import find_executable
from os.path import isfile, join, exists
from ovos_plugin_manager.templates.transformers import TTSTransformer


class SoxTTSTransformer(TTSTransformer):
    """ runs after TTS stage but before playback"""

    def __init__(self, name="ovos-tts-transformer-sox", priority=50, config=None):
        super().__init__(name, priority, config)
        self.default_effects = self.config.get("default_effects", {})

    def transform(self, wav_file: str, context: dict = None) -> str:
        """
        Optionally transform passed wav_file and return path to transformed file
        :param wav_file: path to wav file generated in TTS stage
        :returns: path to transformed wav file for playback
        """
        context = context or {"effects": self.default_effects}
        effects = context.get("effects") or {}
        # effects is a dict of {"effect_name": effect_kwargs}
        mut = TTSMutator(wav_file, effects)
        wav_file = mut.apply(wav_file + ".mutated.wav")
        return wav_file, context


class TTSMutator:
    def __init__(self, sound_file: str, config: dict = None):
        self.config = config or {}
        self.sound_file = sound_file
        sox = find_executable("sox")
        if sox is None:
            raise ImportError("could not find 'sox' executable")
        self.effects = [sox, sound_file, sound_file]

    def apply(self, output=None):
        if len(list(self.config.keys())):
            for effect in self.config:
                params = self.config[effect]
                if effect == "pitch":
                    self.pitch(**params)
                elif effect == "phaser":
                    self.phaser(**params)
                elif effect == "flanger":
                    self.flanger(**params)
                elif effect == "reverb":
                    self.reverb(**params)
                elif effect == "tempo":
                    self.tempo(**params)
                elif effect == "treble":
                    self.treble(**params)
                elif effect == "tremolo":
                    self.tremolo(**params)
                elif effect == "reverse":
                    self.reverse()
                elif effect == "speed":
                    self.speed(**params)
                elif effect == "chorus":
                    self.chorus(**params)
                elif effect == "echo":
                    self.echo(**params)
                elif effect == "bend":
                    self.bend(**params)
                elif effect == "stretch":
                    self.stretch(**params)
                elif effect == "overdrive":
                    self.overdrive(**params)
                elif effect == "bass":
                    self.bass(**params)
                elif effect == "allpass":
                    self.allpass(**params)
                elif effect == "bandpass":
                    self.bandpass(**params)
                elif effect == "bandreject":
                    self.bandreject(**params)
                elif effect == "compand":
                    self.compand(**params)
                elif effect == "contrast":
                    self.contrast(**params)
                elif effect == "equalizer":
                    self.equalizer(**params)
                elif effect == "gain":
                    self.gain(**params)
                elif effect == "highpass":
                    self.highpass(**params)
                elif effect == "lowpass":
                    self.lowpass(**params)
                elif effect == "loudness":
                    self.loudness(**params)
                elif effect == "noisered":
                    self.noisered(**params)
            if output:
                self.save(output)

    def save(self, out_path=None):
        out_path = out_path or self.sound_file
        self.effects[2] = out_path
        subprocess.call(self.effects, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
        return out_path

    def pitch(self, n_semitones, quick=False):
        """
        Pitch shift the audio without changing the tempo.

        This effect uses the WSOLA algorithm. The audio is chopped up into segments which are then shifted in the time domain and overlapped (cross-faded) at points where their waveforms are most similar as determined by measurement of least squares.

        Parameters:
        n_semitones : float
        The number of semitones to shift. Can be positive or negative.

        quick : bool, default=False
        If True, this effect will run faster but with lower sound quality.
        """
        LOG.debug("pitch")
        effect_args = ['pitch']

        if quick:
            effect_args.append('-q')

        effect_args.append('{:f}'.format(n_semitones * 100.))
        self.effects += effect_args

    def phaser(self, gain_in=0.8, gain_out=0.74, delay=3, decay=0.4,
               speed=0.5, modulation_shape='sinusoidal'):
        """
        Apply a phasing effect to the audio.

            Parameters:
            gain_in : float, default=0.8
            Input volume between 0 and 1

            gain_out: float, default=0.74
            Output volume between 0 and 1

            delay : float, default=3
            Delay in miliseconds between 0 and 5

            decay : float, default=0.4
            Decay relative to gain_in, between 0.1 and 0.5.

            speed : float, default=0.5
            Modulation speed in Hz, between 0.1 and 2

            modulation_shape : str, defaul=’sinusoidal’
            Modulation shpae. One of ‘sinusoidal’ or ‘triangular’
        """
        LOG.debug("phaser")
        effect_args = [
            'phaser',
            '{:f}'.format(gain_in),
            '{:f}'.format(gain_out),
            '{:f}'.format(delay),
            '{:f}'.format(decay),
            '{:f}'.format(speed)
        ]

        if modulation_shape == 'sinusoidal':
            effect_args.append('-s')
        elif modulation_shape == 'triangular':
            effect_args.append('-t')
        self.effects += effect_args

    def flanger(self, delay=0, depth=2, regen=0, width=71, speed=0.5,
                shape='sine', phase=25, interp='linear'):
        """
        Apply a flanging effect to the audio.

        Parameters:
        delay : float, default=0
        Base delay (in miliseconds) between 0 and 30.

        depth : float, default=2
        Added swept delay (in miliseconds) between 0 and 10.

        regen : float, default=0
        Percentage regeneration between -95 and 95.

        width : float, default=71,
        Percentage of delayed signal mixed with original between 0 and 100.

        speed : float, default=0.5
        Sweeps per second (in Hz) between 0.1 and 10.

        shape : ‘sine’ or ‘triangle’, default=’sine’
        Swept wave shape

        phase : float, default=25
        Swept wave percentage phase-shift for multi-channel flange between 0 and 100. 0 = 100 = same phase on each channel

        interp : ‘linear’ or ‘quadratic’, default=’linear’
        Digital delay-line interpolation type.
        """
        LOG.debug("flanger")
        effect_args = [
            'flanger',
            '{:f}'.format(delay),
            '{:f}'.format(depth),
            '{:f}'.format(regen),
            '{:f}'.format(width),
            '{:f}'.format(speed),
            '{}'.format(shape),
            '{:f}'.format(phase),
            '{}'.format(interp)
        ]
        self.effects += effect_args

    def reverb(self, reverberance=50, high_freq_damping=50, room_scale=100,
               stereo_depth=100, pre_delay=0, wet_gain=0, wet_only=False):
        """
        Add reverberation to the audio using the ‘freeverb’ algorithm. A reverberation effect is sometimes desirable for concert halls that are too small or contain so many people that the hall’s natural reverberance is diminished. Applying a small amount of stereo reverb to a (dry) mono signal will usually make it sound more natural.

        Parameters:
        reverberance : float, default=50
        Percentage of reverberance

        high_freq_damping : float, default=50
        Percentage of high-frequency damping.

        room_scale : float, default=100
        Scale of the room as a percentage.

        stereo_depth : float, default=100
        Stereo depth as a percentage.

        pre_delay : float, default=0
        Pre-delay in milliseconds.

        wet_gain : float, default=0
        Amount of wet gain in dB

        wet_only : bool, default=False
        If True, only outputs the wet signal.
        """
        LOG.debug("reverb")
        effect_args = ['reverb']

        if wet_only:
            effect_args.append('-w')

        effect_args.extend([
            '{:f}'.format(reverberance),
            '{:f}'.format(high_freq_damping),
            '{:f}'.format(room_scale),
            '{:f}'.format(stereo_depth),
            '{:f}'.format(pre_delay),
            '{:f}'.format(wet_gain)
        ])
        self.effects += effect_args

    def tempo(self, factor, audio_type=None, quick=False):
        """Time stretch audio without changing pitch.

        This effect uses the WSOLA algorithm. The audio is chopped up into segments which are then shifted in the time domain and overlapped (cross-faded) at points where their waveforms are most similar as determined by measurement of least squares.

        Parameters:
        factor : float
        The ratio of new tempo to the old tempo. For ex. 1.1 speeds up the tempo by 10%; 0.9 slows it down by 10%.

        audio_type : str
        Type of audio, which optimizes algorithm parameters. One of:
        m : Music,
        s : Speech,
        l : Linear (useful when factor is close to 1),
        quick : bool, default=False
        If True, this effect will run faster but with lower sound quality.
        """
        LOG.debug("tempo")
        if factor <= 0:
            raise ValueError("factor must be a positive number")

        if factor < 0.5 or factor > 2:
            LOG.warning(
                "Using an extreme time stretching factor. "
                "Quality of results will be poor"
            )

        if abs(factor - 1.0) <= 0.1:
            LOG.warning(
                "For this stretch factor, "
                "the stretch effect has better performance."
            )

        if audio_type not in [None, 'm', 's', 'l']:
            raise ValueError(
                "audio_type must be one of None, 'm', 's', or 'l'."
            )

        if not isinstance(quick, bool):
            raise ValueError("quick must be a boolean.")

        effect_args = ['tempo']

        if quick:
            effect_args.append('-q')

        if audio_type is not None:
            effect_args.append('-{}'.format(audio_type))

        effect_args.append('{:f}'.format(factor))
        self.effects += effect_args

    def treble(self, gain_db, frequency=3000.0, slope=0.5):
        """
        Boost or cut the treble (lower) frequencies of the audio using a two-pole shelving filter with a response similar to that of a standard hi-fi’s tone-controls. This is also known as shelving equalisation.

        The filters are described in detail in http://musicdsp.org/files/Audio-EQ-Cookbook.txt

        Parameters:
        gain_db : float
        The gain at the Nyquist frequency. For a large cut use -20, for a large boost use 20.

        frequency : float, default=100.0
        The filter’s cutoff frequency in Hz.

        slope : float, default=0.5
        The steepness of the filter’s shelf transition. For a gentle slope use 0.3, and use 1.0 for a steep slope.

        """
        LOG.debug("treble")
        effect_args = [
            'treble', '{:f}'.format(gain_db), '{:f}'.format(frequency),
            '{:f}s'.format(slope)
        ]

        self.effects.extend(effect_args)

    def tremolo(self, speed=6.0, depth=40.0):
        """
        Apply a tremolo (low frequency amplitude modulation) effect to the audio. The tremolo frequency in Hz is giv en by speed, and the depth as a percentage by depth (default 40).

        Parameters:
        speed : float
        Tremolo speed in Hz.

        depth : float
        Tremolo depth as a percentage of the total amplitude.

        """
        LOG.debug("tremolo")
        effect_args = [
            'tremolo',
            '{:f}'.format(speed),
            '{:f}'.format(depth)
        ]

        self.effects.extend(effect_args)

    def reverse(self):
        """Reverse the audio completely"""
        LOG.debug("reverse")
        effect_args = ['reverse']
        self.effects.extend(effect_args)

    def speed(self, factor):
        """
        Adjust the audio speed (pitch and tempo together).

        Technically, the speed effect only changes the sample rate information, leaving the samples themselves untouched. The rate effect is invoked automatically to resample to the output sample rate, using its default quality/speed. For higher quality or higher speed resampling, in addition to the speed effect, specify the rate effect with the desired quality option.

        Parameters:
        factor : float
        The ratio of the new speed to the old speed. For ex. 1.1 speeds up the audio by 10%; 0.9 slows it down by 10%. Note - this argument is the inverse of what is passed to the sox stretch effect for consistency with speed.
        """
        LOG.debug("speed: " + str(factor))

        if factor < 0.5 or factor > 2:
            LOG.warning(
                "Using an extreme factor. Quality of results will be poor"
            )

        effect_args = ['speed', '{:f}'.format(factor)]

        self.effects.extend(effect_args)

    def chorus(self, gain_in=0.5, gain_out=0.9, n_voices=3, delays=None,
               decays=None, speeds=None, depths=None, shapes=None):
        """
        Add a chorus effect to the audio. This can makeasingle vocal sound like a chorus, but can also be applied to instrumentation.

        Chorus resembles an echo effect with a short delay, but whereas with echo the delay is constant, with chorus, it is varied using sinusoidal or triangular modulation. The modulation depth defines the range the modulated delay is played before or after the delay. Hence the delayed sound will sound slower or faster, that is the delayed sound tuned around the original one, like in a chorus where some vocals are slightly off key.

        Parameters:
        gain_in : float, default=0.3
        The time in seconds over which the instantaneous level of the input signal is averaged to determine increases in volume.

        gain_out : float, default=0.8
        The time in seconds over which the instantaneous level of the input signal is averaged to determine decreases in volume.

        n_voices : int, default=3
        The number of voices in the chorus effect.

        delays : list of floats > 20 or None, default=None
        If a list, the list of delays (in miliseconds) of length n_voices. If None, the individual delay parameters are chosen automatically to be between 40 and 60 miliseconds.

        decays : list of floats or None, default=None
        If a list, the list of decays (as a fraction of gain_in) of length n_voices. If None, the individual decay parameters are chosen automatically to be between 0.3 and 0.4.

        speeds : list of floats or None, default=None
        If a list, the list of modulation speeds (in Hz) of length n_voices If None, the individual speed parameters are chosen automatically to be between 0.25 and 0.4 Hz.

        depths : list of floats or None, default=None
        If a list, the list of depths (in miliseconds) of length n_voices. If None, the individual delay parameters are chosen automatically to be between 1 and 3 miliseconds.

        shapes : list of ‘s’ or ‘t’ or None, deault=None
        If a list, the list of modulation shapes - ‘s’ for sinusoidal or ‘t’ for triangular - of length n_voices. If None, the individual shapes are chosen automatically.
        """
        LOG.debug("chorus")
        if gain_in <= 0 or gain_in > 1:
            raise ValueError("gain_in must be a number between 0 and 1.")
        if gain_out <= 0 or gain_out > 1:
            raise ValueError("gain_out must be a number between 0 and 1.")
        if not isinstance(n_voices, int) or n_voices <= 0:
            raise ValueError("n_voices must be a positive integer.")

            # validate delays
        if not (delays is None or isinstance(delays, list)):
            raise ValueError("delays must be a list or None")
        if delays is not None:
            if len(delays) != n_voices:
                raise ValueError("the length of delays must equal n_voices")
        else:
            delays = [random.uniform(40, 60) for _ in range(n_voices)]

            # validate decays
        if not (decays is None or isinstance(decays, list)):
            raise ValueError("decays must be a list or None")
        if decays is not None:
            if len(decays) != n_voices:
                raise ValueError("the length of decays must equal n_voices")
        else:
            decays = [random.uniform(0.3, 0.4) for _ in range(n_voices)]

            # validate speeds
        if not (speeds is None or isinstance(speeds, list)):
            raise ValueError("speeds must be a list or None")
        if speeds is not None:
            if len(speeds) != n_voices:
                raise ValueError("the length of speeds must equal n_voices")
        else:
            speeds = [random.uniform(0.25, 0.4) for _ in range(n_voices)]

            # validate depths
        if not (depths is None or isinstance(depths, list)):
            raise ValueError("depths must be a list or None")
        if depths is not None:
            if len(depths) != n_voices:
                raise ValueError("the length of depths must equal n_voices")
        else:
            depths = [random.uniform(1.0, 3.0) for _ in range(n_voices)]

            # validate shapes
        if not (shapes is None or isinstance(shapes, list)):
            raise ValueError("shapes must be a list or None")
        if shapes is not None:
            if len(shapes) != n_voices:
                raise ValueError("the length of shapes must equal n_voices")
            if any((p not in ['t', 's']) for p in shapes):
                raise ValueError("the elements of shapes must be 's' or 't'")
        else:
            shapes = [random.choice(['t', 's']) for _ in range(n_voices)]

        effect_args = ['chorus', '{}'.format(gain_in), '{}'.format(gain_out)]

        for i in range(n_voices):
            effect_args.extend([
                '{:f}'.format(delays[i]),
                '{:f}'.format(decays[i]),
                '{:f}'.format(speeds[i]),
                '{:f}'.format(depths[i]),
                '-{}'.format(shapes[i])
            ])

        self.effects.extend(effect_args)

    def echo(self, gain_in=0.8, gain_out=0.9, n_echos=1, delays=None,
             decays=None):
        """
        Add echoing to the audio.

        Echoes are reflected sound and can occur naturally amongst mountains (and sometimes large buildings) when talking or shouting; digital echo effects emulate this behav- iour and are often used to help fill out the sound of a single instrument or vocal. The time differ- ence between the original signal and the reflection is the ‘delay’ (time), and the loudness of the reflected signal is the ‘decay’. Multiple echoes can have different delays and decays.

        Parameters:
        gain_in : float, default=0.8
        Input volume, between 0 and 1

        gain_out : float, default=0.9
        Output volume, between 0 and 1

        n_echos : int, default=1
        Number of reflections

        delays : list, default=[60]
        List of delays in miliseconds

        decays : list, default=[0.4]
        List of decays, relative to gain in between 0 and 1

        """
        delays = delays or [60]
        decays = decays or [0.4]
        LOG.debug("echo")
        if gain_in <= 0 or gain_in > 1:
            raise ValueError("gain_in must be a number between 0 and 1.")

        if gain_out <= 0 or gain_out > 1:
            raise ValueError("gain_out must be a number between 0 and 1.")

        if not isinstance(n_echos, int) or n_echos <= 0:
            raise ValueError("n_echos must be a positive integer.")

            # validate delays
        if not isinstance(delays, list):
            raise ValueError("delays must be a list")

        if len(delays) != n_echos:
            raise ValueError("the length of delays must equal n_echos")

            # validate decays
        if not isinstance(decays, list):
            raise ValueError("decays must be a list")

        if len(decays) != n_echos:
            raise ValueError("the length of decays must equal n_echos")

        effect_args = ['echo', '{:f}'.format(gain_in), '{:f}'.format(gain_out)]

        for i in range(n_echos):
            effect_args.extend([
                '{}'.format(delays[i]),
                '{}'.format(decays[i])
            ])

        self.effects.extend(effect_args)

    def bend(self, n_bends, start_times, end_times, cents, frame_rate=25,
             oversample_rate=16):
        """
        Changes pitch by specified amounts at specified times. The pitch-bending algorithm utilises the Discrete Fourier Transform (DFT) at a particular frame rate and over-sampling rate.

        Parameters:
        n_bends : int
        The number of intervals to pitch shift

        start_times : list of floats
        A list of absolute start times (in seconds), in order

        end_times : list of floats
        A list of absolute end times (in seconds) in order. [start_time, end_time] intervals may not overlap!

        cents : list of floats
        A list of pitch shifts in cents. A positive value shifts the pitch up, a negative value shifts the pitch down.

        frame_rate : int, default=25
        The number of DFT frames to process per second, between 10 and 80

        oversample_rate: int, default=16
        The number of frames to over sample per second, between 4 and 32
        """
        LOG.debug("bend")
        if not isinstance(n_bends, int) or n_bends < 1:
            raise ValueError("n_bends must be a positive integer.")

        if not isinstance(start_times, list) or len(start_times) != n_bends:
            raise ValueError("start_times must be a list of length n_bends.")

        if any([(p <= 0) for p in start_times]):
            raise ValueError("start_times must be positive floats.")

        if sorted(start_times) != start_times:
            raise ValueError("start_times must be in increasing order.")

        if not isinstance(end_times, list) or len(end_times) != n_bends:
            raise ValueError("end_times must be a list of length n_bends.")

        if any([(p <= 0) for p in end_times]):
            raise ValueError("end_times must be positive floats.")

        if sorted(end_times) != end_times:
            raise ValueError("end_times must be in increasing order.")

        if any([e <= s for s, e in zip(start_times, end_times)]):
            raise ValueError(
                "end_times must be element-wise greater than start_times."
            )

        if any([e > s for s, e in zip(start_times[1:], end_times[:-1])]):
            raise ValueError(
                "[start_time, end_time] intervals must be non-overlapping."
            )

        if not isinstance(cents, list) or len(cents) != n_bends:
            raise ValueError("cents must be a list of length n_bends.")

        if (not isinstance(frame_rate, int) or
                frame_rate < 10 or frame_rate > 80):
            raise ValueError("frame_rate must be an integer between 10 and 80")

        if (not isinstance(oversample_rate, int) or
                oversample_rate < 4 or oversample_rate > 32):
            raise ValueError(
                "oversample_rate must be an integer between 4 and 32."
            )

        effect_args = [
            'bend',
            '-f', '{}'.format(frame_rate),
            '-o', '{}'.format(oversample_rate)
        ]

        last = 0
        for i in range(n_bends):
            t_start = round(start_times[i] - last, 2)
            t_end = round(end_times[i] - start_times[i], 2)
            effect_args.append(
                '{:f},{:f},{:f}'.format(t_start, cents[i], t_end)
            )
            last = end_times[i]

        self.effects.extend(effect_args)

    def stretch(self, factor, window=20):
        """
        Change the audio duration (but not its pitch). Unless factor is close to 1, use the tempo effect instead.

        This effect is broadly equivalent to the tempo effect with search set to zero, so in general, its results are comparatively poor; it is retained as it can sometimes out-perform tempo for small factors.

        Parameters:
        factor : float
        The ratio of the new tempo to the old tempo. For ex. 1.1 speeds up the tempo by 10%; 0.9 slows it down by 10%. Note - this argument is the inverse of what is passed to the sox stretch effect for consistency with tempo.

        window : float, default=20
        Window size in miliseconds
        """
        LOG.debug("stretch")
        if factor <= 0:
            raise ValueError("factor must be a positive number")

        if factor < 0.5 or factor > 2:
            LOG.warning(
                "Using an extreme time stretching factor. "
                "Quality of results will be poor"
            )

        if abs(factor - 1.0) > 0.1:
            LOG.warning(
                "For this stretch factor, "
                "the tempo effect has better performance."
            )

        if window <= 0:
            raise ValueError(
                "window must be a positive number."
            )

        effect_args = ['stretch', '{:f}'.format(factor), '{:f}'.format(window)]

        self.effects.extend(effect_args)

    def overdrive(self, gain_db=20.0, colour=20.0):
        """
        Apply non-linear distortion.

        Parameters:
        gain_db : float, default=20
        Controls the amount of distortion (dB).

        colour : float, default=20
        Controls the amount of even harmonic content in the output (dB).
        """
        LOG.debug("overdrive")
        effect_args = [
            'overdrive',
            '{:f}'.format(gain_db),
            '{:f}'.format(colour)
        ]
        self.effects.extend(effect_args)

    def bass(self, gain_db, frequency=100.0, slope=0.5):
        """
        Boost or cut the bass (lower) frequencies of the audio using a two-pole shelving filter with a response similar to that of a standard hi-fi’s tone-controls. This is also known as shelving equalisation.

        The filters are described in detail in http://musicdsp.org/files/Audio-EQ-Cookbook.txt

        Parameters:
        gain_db : float
        The gain at 0 Hz. For a large cut use -20, for a large boost use 20.

        frequency : float, default=100.0
        The filter’s cutoff frequency in Hz.

        slope : float, default=0.5
        The steepness of the filter’s shelf transition. For a gentle slope use 0.3, and use 1.0 for a steep slope.
        """
        LOG.debug("bass")
        if frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        effect_args = [
            'bass', '{:f}'.format(gain_db), '{:f}'.format(frequency),
            '{:f}s'.format(slope)
        ]

        self.effects.extend(effect_args)

    def allpass(self, frequency, width_q=2.0):
        """
        Apply a two-pole all-pass filter. An all-pass filter changes the audio’s frequency to phase relationship without changing its frequency to amplitude relationship. The filter is described in detail in at http://musicdsp.org/files/Audio-EQ-Cookbook.txt

        Parameters:
        frequency : float
        The filter’s center frequency in Hz.

        width_q : float, default=2.0
        The filter’s width as a Q-factor.
        """
        LOG.debug("allpass")
        if frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        effect_args = [
            'allpass', '{:f}'.format(frequency), '{:f}q'.format(width_q)
        ]

        self.effects.extend(effect_args)

    def bandpass(self, frequency, width_q=2.0, constant_skirt=False):
        """
        Apply a two-pole Butterworth band-pass filter with the given central frequency, and (3dB-point) band-width. The filter rolls off at 6dB per octave (20dB per decade) and is described in detail in http://musicdsp.org/files/Audio-EQ-Cookbook.txt
        Parameters:
        frequency : float
        The filter’s center frequency in Hz.

        width_q : float, default=2.0
        The filter’s width as a Q-factor.

        constant_skirt : bool, default=False
        If True, selects constant skirt gain (peak gain = width_q). If False, selects constant 0dB peak gain.
        """
        LOG.debug("bandpass")
        if frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        if not isinstance(constant_skirt, bool):
            raise ValueError("constant_skirt must be a boolean.")

        effect_args = ['bandpass']

        if constant_skirt:
            effect_args.append('-c')

        effect_args.extend(['{:f}'.format(frequency), '{:f}q'.format(width_q)])

        self.effects.extend(effect_args)

    def bandreject(self, frequency, width_q=2.0):
        """
        Apply a two-pole Butterworth band-reject filter with the given central frequency, and (3dB-point) band-width. The filter rolls off at 6dB per octave (20dB per decade) and is described in detail in http://musicdsp.org/files/Audio-EQ-Cookbook.txt

        Parameters:
        frequency : float
        The filter’s center frequency in Hz.

        width_q : float, default=2.0
        The filter’s width as a Q-factor.

        constant_skirt : bool, default=False
        If True, selects constant skirt gain (peak gain = width_q). If False, selects constant 0dB peak gain.
        """
        LOG.debug("bandreject")
        if frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        effect_args = [
            'bandreject', '{:f}'.format(frequency), '{:f}q'.format(width_q)
        ]

        self.effects.extend(effect_args)

    def compand(self, attack_time=0.3, decay_time=0.8, soft_knee_db=6.0,
                tf_points=None):
        """
        Compand (compress or expand) the dynamic range of the audio.

        Parameters:
        attack_time : float, default=0.3
        The time in seconds over which the instantaneous level of the input signal is averaged to determine increases in volume.

        decay_time : float, default=0.8
        The time in seconds over which the instantaneous level of the input signal is averaged to determine decreases in volume.

        soft_knee_db : float or None, default=6.0
        The ammount (in dB) for which the points at where adjacent line segments on the transfer function meet will be rounded. If None, no soft_knee is applied.

        tf_points : list of tuples
        Transfer function points as a list of tuples corresponding to points in (dB, dB) defining the compander’s transfer function.
        """
        tf_points = tf_points or [(-70, -70), (-60, -20), (0, 0)]
        LOG.debug("compand")
        if attack_time <= 0:
            raise ValueError("attack_time must be a positive number.")

        if decay_time <= 0:
            raise ValueError("decay_time must be a positive number.")

        if attack_time > decay_time:
            LOG.warning(
                "attack_time is larger than decay_time.\n"
                "For most situations, attack_time should be shorter than "
                "decay time because the human ear is more sensitive to sudden "
                "loud music than sudden soft music."
            )

        if not isinstance(tf_points, list):
            raise TypeError("tf_points must be a list.")
        if len(tf_points) == 0:
            raise ValueError("tf_points must have at least one point.")
        if any(not isinstance(pair, tuple) for pair in tf_points):
            raise ValueError("elements of tf_points must be pairs")
        if any(len(pair) != 2 for pair in tf_points):
            raise ValueError("Tuples in tf_points must be length 2")
        if any((p[0] > 0 or p[1] > 0) for p in tf_points):
            raise ValueError("Tuple values in tf_points must be <= 0 (dB).")
        if len(tf_points) > len(set([p[0] for p in tf_points])):
            raise ValueError("Found duplicate x-value in tf_points.")

        tf_points = sorted(
            tf_points,
            key=lambda tf_points: tf_points[0]
        )
        transfer_list = []
        for point in tf_points:
            transfer_list.extend([
                "{:f}".format(point[0]), "{:f}".format(point[1])
            ])

        effect_args = [
            'compand',
            "{:f},{:f}".format(attack_time, decay_time)
        ]

        if soft_knee_db is not None:
            effect_args.append(
                "{:f}:{}".format(soft_knee_db, ",".join(transfer_list))
            )
        else:
            effect_args.append(",".join(transfer_list))

        self.effects.extend(effect_args)

    def contrast(self, amount=75):
        """
        Comparable with compression, this effect modifies an audio signal to make it sound louder.

        Parameters:
        amount : float
        Amount of enhancement between 0 and 100.

        """
        LOG.debug("contrast")
        if amount < 0 or amount > 100:
            raise ValueError('amount must be a number between 0 and 100.')

        effect_args = ['contrast', '{:f}'.format(amount)]

        self.effects.extend(effect_args)

    def equalizer(self, frequency, width_q, gain_db):
        """
        Apply a two-pole peaking equalisation (EQ) filter to boost or reduce around a given frequency. This effect can be applied multiple times to produce complex EQ curves.

        Parameters:
        frequency : float
        The filter’s central frequency in Hz.

        width_q : float
        The filter’s width as a Q-factor.

        gain_db : float
        The filter’s gain in dB.
        """
        LOG.debug("equalizer")
        if frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        effect_args = [
            'equalizer',
            '{:f}'.format(frequency),
            '{:f}q'.format(width_q),
            '{:f}'.format(gain_db)
        ]
        self.effects.extend(effect_args)

    def gain(self, gain_db=0.0, normalize=True, limiter=False, balance=None):
        """
        Apply amplification or attenuation to the audio signal.

        Parameters:
        gain_db : float, default=0.0
        Gain adjustment in decibels (dB).

        normalize : bool, default=True
        If True, audio is normalized to gain_db relative to full scale. If False, simply adjusts the audio power level by gain_db.

        limiter : bool, default=False
        If True, a simple limiter is invoked to prevent clipping.

        balance : str or None, default=None
        Balance gain across channels. Can be one of:
        None applies no balancing (default)
        ‘e’ applies gain to all channels other than that with the
        highest peak level, such that all channels attain the same peak level
        ‘B’ applies gain to all channels other than that with the
        highest RMS level, such that all channels attain the same RMS level
        ‘b’ applies gain with clipping protection to all channels other
        than that with the highest RMS level, such that all channels attain the same RMS level
        If normalize=True, ‘B’ and ‘b’ are equivalent.
        """
        LOG.debug("gain")
        if balance not in [None, 'e', 'B', 'b']:
            raise ValueError("balance must be one of None, 'e', 'B', or 'b'.")

        effect_args = ['gain']

        if balance is not None:
            effect_args.append('-{}'.format(balance))

        if normalize:
            effect_args.append('-n')

        if limiter:
            effect_args.append('-l')

        effect_args.append('{:f}'.format(gain_db))
        self.effects.extend(effect_args)

    def highpass(self, frequency, width_q=0.707, n_poles=2):
        """
        Apply a high-pass filter with 3dB point frequency. The filter can be either single-pole or double-pole. The filters roll off at 6dB per pole per octave (20dB per pole per decade).

        Parameters:
        frequency : float
        The filter’s cutoff frequency in Hz.

        width_q : float, default=0.707
        The filter’s width as a Q-factor. Applies only when n_poles=2. The default gives a Butterworth response.

        n_poles : int, default=2
        The number of poles in the filter. Must be either 1 or 2
        """
        LOG.debug("highpass")
        if frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        if n_poles not in [1, 2]:
            raise ValueError("n_poles must be 1 or 2.")

        effect_args = [
            'highpass', '-{}'.format(n_poles), '{:f}'.format(frequency)
        ]

        if n_poles == 2:
            effect_args.append('{:f}q'.format(width_q))

        self.effects.extend(effect_args)

    def lowpass(self, frequency, width_q=0.707, n_poles=2):
        """
        Apply a low-pass filter with 3dB point frequency. The filter can be either single-pole or double-pole.
        The filters roll off at 6dB per pole per octave (20dB per pole per decade).

        Parameters:
        frequency : float
        The filter’s cutoff frequency in Hz.

        width_q : float, default=0.707
        The filter’s width as a Q-factor. Applies only when n_poles=2. The default gives a Butterworth response.

        n_poles : int, default=2
        The number of poles in the filter. Must be either 1 or 2
        """
        LOG.debug("lowpass")
        if frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        if n_poles not in [1, 2]:
            raise ValueError("n_poles must be 1 or 2.")

        effect_args = [
            'lowpass', '-{}'.format(n_poles), '{:f}'.format(frequency)
        ]

        if n_poles == 2:
            effect_args.append('{:f}q'.format(width_q))

        self.effects.extend(effect_args)

    def loudness(self, gain_db=-10.0, reference_level=65.0):
        """
        Loudness control. Similar to the gain effect, but provides equalisation for the human auditory system.

        The gain is adjusted by gain_db and the signal is equalised according to ISO 226 w.r.t. reference_level.

        Parameters:
        gain_db : float, default=-10.0
        Loudness adjustment amount (in dB)

        reference_level : float, default=65.0
        Reference level (in dB) according to which the signal is equalized. Must be between 50 and 75 (dB)
        """
        LOG.debug("loudness")
        if reference_level > 75 or reference_level < 50:
            raise ValueError('reference_level must be between 50 and 75')

        effect_args = [
            'loudness',
            '{:f}'.format(gain_db),
            '{:f}'.format(reference_level)
        ]
        self.effects.extend(effect_args)

    def noisered(self, profile_path, amount=0.5):
        """
        Reduce noise in the audio signal by profiling and filtering. This effect is moderately effective at removing consistent background noise such as hiss or hum.

        Parameters:
        profile_path : str
        Path to a noise profile file. This file can be generated using the noiseprof effect.

        amount : float, default=0.5
        How much noise should be removed is specified by amount. Should be between 0 and 1. Higher numbers will remove more noise but present a greater likelihood of removing wanted components of the audio signal.
        """
        # TODO auto gen profile file
        LOG.info("noisered")
        if not exists(profile_path):
            raise IOError(
                "profile_path {} does not exist.".format(profile_path))

        if amount < 0 or amount > 1:
            raise ValueError("amount must be a number between 0 and 1.")

        effect_args = [
            'noisered',
            profile_path,
            '{:f}'.format(amount)
        ]
        self.effects.extend(effect_args)
