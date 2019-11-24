# JTinn

`JTinn` is a Java port of [Tinn - The tiny neural network library](https://github.com/glouw/tinn).

## Supported features

`JTinn` aims to remain compatible with `tinn`'s API as much as possible, as
well as its serialization semantics. The core `tinn` api is exposed under the
`XTinn` class.

During serialization/deserialization of saved data, `JTinn` will
also save the hidden to output layer weights, in addition to the data saved by
default by `tinn`, which should make a `JTinn` trained network loadable
by `tinn`.

The reference test case replicates written character recognition from the same
data set as `tinn`.

Binary artifacts are available on JCenter at the following coordinates:

```xml
<dependency>
  <groupId>io.vacco.jtinn</groupId>
  <artifactId>jtinn</artifactId>
  <version>1.0.1</version>
</dependency>
```

## API extensions

Whereas `tinn`'s api works as intended, I think it would be useful to extend
the core api under a `JTinn` namespace to support customizations such as:

- Multilayer support.
- Extended activation function support.
- Per layer activation function assignments.

### Disclaimer

> This project is not production ready, and still requires security and code
> correctness audits. You use this software at your own risk.
> Vaccove Crana, LLC., its affiliates and subsidiaries waive any and all
> liability for any damages caused to you by your usage of this software.
