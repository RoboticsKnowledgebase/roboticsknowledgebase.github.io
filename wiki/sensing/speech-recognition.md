---
title: Speech Recognition
---
Speech Recognition can be a very efficient and powerful way to interface with a robot. It has its downsides since it requires a parser to do something intelligent and be intuitive to use. Also, it is not the most reliable mode of communication since it is plagued with background noise in a regular environment and is prone to false matches.

## Resources
Here are a few resources to implement speech recognition:
- Offline
  - Among the various options available online, CMU Sphinx is the most versatile and actively supported open source speech recognition system. It gives the user a low level access to the software. You can create custom dictionaries, train it with custom data set, get confidence scores and use C++, Java or Python for development. To get started you can go through the documentation provided here: http://cmusphinx.sourceforge.net/wiki/
  - Many acoustic and language models are available for download over [here](https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/)
  - Even though these might not perform great, they still provide a solid starting point. One can implement many filtration techniques to get the best output by utilizing confidence scores, setting custom dictionaries and sentence structures.
- Online
  - If internet connectivity and uploading data online is not an issue, Google's Speech API massively outperforms any offline system with the only downside being that there is no low level access so the only thing you can do is upload an audio file and get back a string of output.
  - You can make use of [Python's Speech Recognition Package](https://pypi.python.org/pypi/SpeechRecognition/), to use this API and many others. This package also supports CMUSphinx.

> Note: Google's Speech API requires a key which can be found in the source code of this package. You can bypass the use of this entire package if you don't wish to have this additional layer between Google's Speech API and your script.
The source code with examples can be found [here.](https://github.com/Uberi/speech_recognition)
The key can be found [here.](https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py#L613)
