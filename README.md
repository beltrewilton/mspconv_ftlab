<h2> Estructura </h2>
<div>
<pre class="notranslate">
├── data/                                                      
│   └── MSPCORPUS                                              // dataset de MSP
│       ├── Annotations
│       │   ├── Arousal
│       │   │   └── .csv files for arousal audio label 
│       │   ├── Dominance
│       │   │   └── .csv files for dominance audio label 
│       │   ├── Valence
│       │       └── .csv files for valence audio label 
│       ├── Audio
│       │   └── .wav files containing all the audios 
│       ├── Speaker_Diarization
│       │   └── .json files with speaker numbering 
│       └── Time_Labels
│           └── conversation_parts.txt
│           └── conversation.txt    
├── notebooks/                                                 // notebooks con código y experimentos
└── src/                                                       // archivos con las funciones usadas para levantar los modelos
</pre>
</div>
