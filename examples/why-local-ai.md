# Why run AI on your own PC?

Most AI assistants live in a data center. Every question, document and voice note you give them
travels across the internet, is processed on someone else's computer, and is billed by the token.

Local AI flips that around. The model files sit on your own disk, and your own processor does the
work. Nothing you type or say has to leave the machine, it keeps working without an internet
connection, and there is no per-request bill.

Modern laptops are ready for it. A Ryzen AI processor combines three kinds of engines: the CPU for
general work, a Radeon GPU for heavy parallel math such as image generation, and an NPU, a small
processor built for efficient AI inference.

Lemonade is an open source server that runs language, image, speech and transcription models on that
hardware. Apps talk to it through the same kind of API they would use for a cloud service, so
switching from cloud to local is often a one-line change.

Lemonade Omni models bundle a language model, an image model, a speech recognizer and a voice into
one package. One download gives an app the ability to read, write, draw, listen and speak.

The result is private AI that schools, clinics, small businesses and hobbyists can run on hardware they
already own.
