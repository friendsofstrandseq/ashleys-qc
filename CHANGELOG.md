## v0.2.0dev

Running list of changes:

- clean up logging in all modules, and suppress `logging` command line argument
- add noise levels `verbose` and `debug` to enable logging to `stdout` and `stderr`, respectively
   - default behavior: silent, log warnings and above to `stderr`
- fix [gh#13](https://github.com/friendsofstrandseq/ashleys-qc/issues/13#issue-876048473): fail if no BAM input files can be collected for feature generation
- reactivate `version` option for command line

## v0.2.0

Published version, compatible with `ashleys-qc-pipeline` as-is.