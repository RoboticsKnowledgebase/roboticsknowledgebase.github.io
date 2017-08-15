---
title: Python Construct Library
---

Sending data over serial communication channels in robotic systems that need real time updates between multiple agents calls for ensuring reliability and integrity. Construct is a Python library that comes to the rescue for building and parsing binary data efficiently without the need for writing unnecessary imperative code to build message packets.

The Python Construct is a powerful declarative parser (and builder) for binary data. It declaratively defines a data structure that describes the data. It can be used to parse data into python data structures or to convert these data structures into binary data to be sent over the channel. It is extremely easy to use after you install all the required dependencies into your machine.

## Features
Some key features of Construct are:
- Bit and byte granularity
- Easy to extend subclass system
- Fields: raw bytes or numerical types
- Structs and Sequences: combine simpler constructs into more complex ones
- Adapters: change how data is represented
- Arrays/Ranges: duplicate constructs
- Meta-constructs: use the context (history) to compute the size of data
- If/Switch: branch the computational path based on the context
- On-demand (lazy) parsing: read only what you require
- Pointers: jump from here to there in the data stream

You might not need to use all of the above features if the data you need to send and receive is a simple list of say, waypoints or agent IDs. But it is worth exploring the possible extent of complexity of data that this library can handle. The library provides both simple, atomic constructs (UBINT8, UBNIT16, etc), as well as composite ones which allow you form hierarchical structures of increasing complexity.

## Example Usage
This tool could especially come in handy if your data has many different kinds of fields. As an example, consider a message that contains the agent ID, message type (defined by a word, such as ‘Request’ or ‘Action’), flags and finally the data field that contains a list of integers. Building and parsing such a message would require many lines of code which can be avoided by simply using Construct and defining these fields in a custom format. An example message format is given below.

### Example format
```
message_crc = Struct('message_crc', UBInt64('crc'))

message_format = Struct('message_format',
    ULInt16('vcl_id'),
    Enum(Byte('message_type'),
    HELLO = 0x40,
    INTRO = 0x3f,
    UPDATE = 0x30,
    GOODBYE = 0x20,
    PARKED = 0x31,

        _default_ = Pass
    ),
    Byte('datalen'),
    Array(lambda ctx: ctx['datalen'], Byte('data')),
    Embed(message_crc)
)


if __name__ == "__main__":
    raw = message_format.build(Container(
    vcl_id=0x1,
        message_type='HELLO',
        datalen=4,
        data=[0x1, 0xff, 0xff, 0xdd],
        crc=0x12345678))

    print raw
    mymsg=raw.encode('hex')
    print mymsg
    x=message_format.parse(raw)
    print x
```

The CRC (Cyclic Redundancy Check) used in this snippet is an error-detecting code commonly used in digital networks to detect accidental changes to raw data. Blocks of data get a short check value attached usually at the end, based on the remainder of a polynomial division of their contents. On retrieval, the calculation is repeated and, in the event the check values do not match, exceptions can be handles in your code to take corrective action. This is a very efficient means to check validity of data received, which can be crucial to avoid errors in operation of real time systems.

There are a number of possible methods to check data integrity. CRC checks like CRC11, CRC12, CRC32, etc are commonly used error checking codes that can be used. If you face an issue with using CRC of varying lengths with your data, try using cryptographic hash functions like MD5, which might solve the problem. Further reading can be found [here on StackOverflow.](http://stackoverflow.com/questions/16122067/md5-vs-crc32-which-ones-better-for-common-use)

The snippets (python) below serve as examples of how to build and recover messages using Construct:
```
def build_msg(vcl_id, message_type):

    data = [0x1,0xff,0xff,0xdd]

    datalen = len(data)

    raw = message_format.build(Container(
        vcl_id = vcl_id,
        message_type = message_type,
        datalen = datalen,
        data = data,
        crc = 0))

    msg_without_crc = raw[:-8]
    msg_crc = message_crc.build(Container(
        crc = int(''.join([i for i in hashlib.md5(msg_without_crc).hexdigest() if i.isdigit()])[0:10])))

    msg = msg_without_crc + msg_crc

    pw = ProtocolWrapper(
            header = PROTOCOL_HEADER,
            footer = PROTOCOL_FOOTER,
            dle = PROTOCOL_DLE)
    return pw.wrap(msg)
```
```
def recover_msg(msg):
        pw = ProtocolWrapper(
        header = PROTOCOL_HEADER,
        footer = PROTOCOL_FOOTER,
        dle = PROTOCOL_DLE)

        status = map(pw.input, msg)
        rec_crc = 0
        calc_crc = 1

        if status[-1] == ProtocolStatus.MSG_OK:
                rec_msg = pw.last_message
                rec_crc = message_crc.parse(rec_msg[-8:]).crc
                calc_crc = int(''.join([i for i in hashlib.md5(rec_msg[:-8]).hexdigest() if i.isdigit()])[0:10])

        if rec_crc != calc_crc:
            print 'Error: CRC mismatch'
            return None
        else:
            return rec_msg
```

The following are a guidelines to use Python Construct:
1. Download the suitable version of Construct [here.](https://pypi.python.org/pypi/construct)
2. Copy over this folder to the appropriate code directory. If using ROS, it should be inside your ROS package)
3. You need two additional files `myformat.py` and `protocolwrapper.py` to get started. These can be found [here.](http://eli.thegreenplace.net/2009/08/20/frames-and-protocols-for-the-serial-port-in-python) This is a great resource for example code and supporting protocol wrappers to be used for serial communication (in Python)

## Resources
- Construct’s homepage is http://construct.readthedocs.org/ where you can find all kinds of docs and resources.
- The library itself is developed on https://github.com/construct/construct.
- For discussion and queries, here is a link to the [Google group](https://groups.google.com/forum/#!forum/construct3).
- Construct should run on any Python 2.5-3.3 implementation. Its only requirement is [six](http://pypi.python.org/pypi/six), which is used to overcome the differences between Python 2 and 3.
