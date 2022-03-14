# CRAFT: Constraining and Restoring iterative Algorithm for Faraday Tomography

A python 3 implementation of the CRAFT technique described in [Cooray et al. 2020](https://doi.org/10.1093/mnras/staa3580). The techniques takes in an observed complex linear polarization spectrum that is limited in frequency to produce a reconstructed spectrum. If you make use of this code, please cite the above mentioned paper.

### Installation

Install using pip:
```
pip install craft-reconstruct
```
Or from Github repo:
```
pip install git+https://github.com/suchethac/craft
```

Or, clone this directory and install locally:
```
git clone https://github.com/suchethac/craft
cd craft/
pip install .
# - OR -
python setup.py install
```

### Usage

Use as a module by `import craft`.

For learning how to use this code, I have uploaded a demo python notebook. Additional documentation will be available in due time.

If you have any questions, please contact Suchetha Cooray at cooray{at}nagoya-u.jp

### Reference
Suchetha Cooray, Tsutomu T Takeuchi, Takuya Akahori, Yoshimitsu Miyashita, Shinsuke Ideguchi, Keitaro Takahashi, Kiyotomo Ichiki, *An Iterative Reconstruction Algorithm for Faraday Tomography, Monthly Notices of the Royal Astronomical Society*, Volume 500, Issue 4, February 2021, Pages 5129–5141, <https://doi.org/10.1093/mnras/staa3580>
