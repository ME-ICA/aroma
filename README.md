# Aroma

[![codecov](https://codecov.io/gh/ME-ICA/aroma/branch/main/graph/badge.svg)](https://codecov.io/gh/ME-ICA/aroma)
[![CircleCI](https://circleci.com/gh/ME-ICA/aroma.svg?branch=main&style=shield)](https://circleci.com/gh/ME-ICA/aroma) [![Join the chat at https://gitter.im/ME-ICA/aroma](https://badges.gitter.im/ME-ICA/aroma.svg)](https://gitter.im/ME-ICA/aroma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## License info

This repository is a clone of [maartenmennes/ICA-AROMA](https://github.com/maartenmennes/ICA-AROMA),
which has an Apache 2.0 license. We plan to modify this clone in order to restructure it into a
pure Python package, but we will make sure to fulfill the requirements of the license.

#### Source of the Gaussian-Gamma Mixture Model

The Gaussian-Gamma Mixture Model used in this package (`aroma.mixture.GGM`) was taken from
the [`nipy`](https://github.com/nipy/nipy) package. As such, the file in which the model
is defined (`aroma/mixture.py`) falls under `nipy`'s licensing. The license for that file
is stored in the file's docstring.
