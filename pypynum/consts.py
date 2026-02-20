"""
This module defines a collection of constants representing various physical, mathematical, and unit conversion
factors. These constants are commonly used in scientific and engineering calculations.

- **EB**: Exabyte, equal to 2^60 bytes.
- **G**: Gravitational constant (m^3 kg^-1 s^-2).
- **GB**: Gigabyte, equal to 2^30 bytes.
- **KB**: Kilobyte, equal to 2^10 bytes.
- **MB**: Megabyte, equal to 2^20 bytes.
- **PB**: Petabyte, equal to 2^50 bytes.
- **TB**: Terabyte, equal to 2^40 bytes.
- **YB**: Yottabyte, equal to 2^80 bytes.
- **ZB**: Zettabyte, equal to 2^70 bytes.
- **a_0**: Bohr radius (meters).
- **acre**: Area of one acre (square meters).
- **alpha**: Fine-structure constant.
- **alpha_f**: Feigenbaum constant.
- **amu**: Atomic mass unit (kilograms).
- **apery**: Apery's constant.
- **arcmin**: Minute of arc (degrees).
- **arcsec**: Second of arc (degrees).
- **atm**: Standard atmosphere (pascals).
- **atto**: Prefix denoting 10^-18.
- **au**: Astronomical unit (meters).
- **btu**: British thermal unit (joules).
- **c**: Speed of light in a vacuum (meters per second).
- **cal**: Calorie (joules).
- **catalan**: Catalan's constant.
- **centi**: Prefix denoting 10^-2.
- **ci**: Curie (becquerels).
- **conway**: Conway's constant.
- **ct**: Carat (kilograms).
- **cup**: Volume of one US cup (cubic meters).
- **day**: Number of seconds in one day.
- **debye**: Unit of electric dipole moment (coulomb-meters).
- **deci**: Prefix denoting 10^-1.
- **deka**: Prefix denoting 10^1.
- **delta_f**: Feigenbaum constant delta.
- **e**: Euler's number.
- **epsilon_0**: Vacuum permittivity (farads per meter).
- **exa**: Prefix denoting 10^18.
- **faraday**: Faraday constant (coulombs).
- **fathom**: Length of one fathom (meters).
- **fc**: Foot-candle (lumens per square meter).
- **femto**: Prefix denoting 10^-15.
- **fl**: Foot-lambert (candelas per square meter).
- **fl_oz**: Fluid ounce (cubic meters).
- **foias**: Foias constant.
- **foot**: Length of one foot (meters).
- **g**: Standard acceleration due to gravity (m/s^2).
- **gal_can**: Canadian gallon (cubic meters).
- **gal_uk**: UK gallon (cubic meters).
- **gal_us**: US gallon (cubic meters).
- **gamma**: Euler-Mascheroni constant.
- **gauss**: Unit of magnetic field strength (tesla).
- **giga**: Prefix denoting 10^9.
- **glaisher**: Glaisher-Kinkelin constant.
- **h**: Planck constant (joule-seconds).
- **hbar**: Reduced Planck constant (joule-seconds).
- **hecto**: Prefix denoting 10^2.
- **hour**: Number of seconds in one hour.
- **hp**: Horsepower (watts).
- **inch**: Length of one inch (meters).
- **inf**: Infinity (float representation).
- **inh2o**: Inches of water (pascals).
- **inhg**: Inches of mercury (pascals).
- **k_b**: Boltzmann constant (joules per kelvin).
- **k_e**: Coulomb's constant (newton-meter^2 per coulomb^2).
- **kilo**: Prefix denoting 10^3.
- **kip**: Kip (force) (newtons).
- **kmh**: Kilometers per hour (meters per second).
- **knot**: Nautical mile per hour (meters per second).
- **lbf**: Pound-force (newtons).
- **lbm**: Pound-mass (kilograms).
- **ly**: Light-year (meters).
- **m_e**: Electron mass (kilograms).
- **m_mu**: Muon mass (kilograms).
- **m_n**: Neutron mass (kilograms).
- **m_p**: Proton mass (kilograms).
- **m_sun**: Solar mass (kilograms).
- **mega**: Prefix denoting 10^6.
- **micro**: Prefix denoting 10^-6.
- **mil**: Thousandth of an inch (meters).
- **mile**: Length of one mile (meters).
- **milli**: Prefix denoting 10^-3.
- **minute**: Number of seconds in one minute.
- **mmhg**: Millimeters of mercury (pascals).
- **mph**: Miles per hour (meters per second).
- **mu_0**: Vacuum permeability (henry per meter).
- **mu_b**: Bohr magneton (joules per tesla).
- **mu_e**: Electron magnetic moment (joules per tesla).
- **mu_n**: Neutron magnetic moment (joules per tesla).
- **mu_p**: Proton magnetic moment (joules per tesla).
- **n_a**: Avogadro constant (per mole).
- **nan**: Not a number (float representation).
- **nano**: Prefix denoting 10^-9.
- **nmi**: Nautical mile (meters).
- **oz_t**: Troy ounce (kilograms).
- **ozm**: Ounce-mass (kilograms).
- **parsec**: Parsec (meters).
- **pdl**: Poundal (force) (newtons).
- **peta**: Prefix denoting 10^15.
- **phi**: Golden ratio.
- **pi**: Pi, the ratio of a circle's circumference to its diameter.
- **pico**: Prefix denoting 10^-12.
- **pint**: Volume of one pint (cubic meters).
- **point**: Point (typography) (meters).
- **psi**: Pounds per square inch (pascals).
- **q_e**: Elementary charge (coulombs).
- **quart**: Volume of one quart (cubic meters).
- **r_g**: Universal gas constant (joules per mole kelvin).
- **radian**: Radians in one degree.
- **roentgen**: Roentgen (exposure to ionizing radiation).
- **ry**: Rydberg constant (energy).
- **sigma**: Stefan-Boltzmann constant (watts per square meter per kelvin^4).
- **sigma_t**: Thomson cross-section (square meters).
- **tbsp**: Tablespoon (cubic meters).
- **tera**: Prefix denoting 10^12.
- **texpoint**: Tex point (typography) (meters).
- **therm**: Therm (energy).
- **torr**: Torr (pressure).
- **tsp**: Teaspoon (cubic meters).
- **uk_ton**: UK ton (kilograms).
- **us_ton**: US ton (kilograms).
- **v_m**: Molar volume (cubic meters per mole).
- **week**: Number of seconds in one week.
- **yard**: Length of one yard (meters).
- **year**: Number of seconds in one year.
- **yocto**: Prefix denoting 10^-24.
- **yotta**: Prefix denoting 10^24.
- **zepto**: Prefix denoting 10^-21.
- **zetta**: Prefix denoting 10^21.
"""

EB = 1152921504606846976
G = 6.6743e-11
GB = 1073741824
KB = 1024
MB = 1048576
PB = 1125899906842624
TB = 1099511627776
YB = 1208925819614629174706176
ZB = 1180591620717411303424
a_0 = 5.291772083e-11
acre = 4046.85642241
alpha = 0.0072973525693
alpha_f = 2.5029078750958926
amu = 1.6605402e-27
apery = 1.2020569031595942
arcmin = 0.016666666666666666
arcsec = 0.0002777777777777778
atm = 101325
atto = 1e-18
au = 149597870700
btu = 1055.05585262
c = 299792458
cal = 4.1868
catalan = 0.915965594177219
centi = 0.01
ci = 37000000000
conway = 1.3035772690342964
ct = 0.0002
cup = 0.000236588236501
day = 86400
debye = 3.33564095198e-30
deci = 0.1
deka = 10
delta_f = 4.66920160910299
e = 2.718281828459045
epsilon_0 = 8.854187817620389e-12
exa = 1000000000000000000
faraday = 96485.3429775
fathom = 1.8288
fc = 10.76
femto = 1e-15
fl = 10.7639104
fl_oz = 2.95735295626e-05
foias = 1.1874523510652712
foot = 0.3048
g = 9.80665
gal_can = 0.00454609
gal_uk = 0.004546092
gal_us = 0.00378541178402
gamma = 0.5772156649015329
gauss = 0.8346268416740732
giga = 1000000000
glaisher = 1.2824271291006226
h = 6.62607015e-34
hbar = 1.0545718176461565e-34
hecto = 100
hour = 3600
hp = 745.6998715822701
inch = 0.0254
inf = float("inf")
inh2o = 249.0889
inhg = 3386.38815789
k_b = 1.380649e-23
k_e = 8.9875517923e+9
kilo = 1000
kip = 4448.22161526
kmh = 0.2777777777777778
knot = 0.5144444444444445
lbf = 4.44822161526
lbm = 0.45359237
ly = 9460730472580800
m_e = 9.1093837015e-31
m_mu = 1.88353109e-28
m_n = 1.67492749804e-27
m_p = 1.67262192369e-27
m_sun = 1.98892e+30
mega = 1000000
micro = 1e-06
mil = 2.54e-05
mile = 1609.344
milli = 0.001
minute = 60
mmhg = 133322.368421
mph = 0.44704
mu_0 = 1.2566370614359173e-06
mu_b = 9.27400899e-24
mu_e = 9.28476362e-24
mu_n = 5.05078317e-27
mu_p = 1.410606633e-26
n_a = 6.02214076e+23
nan = float("nan")
nano = 1e-09
nmi = 1852
oz_t = 0.031103475
ozm = 0.028349523125
parsec = 3.085677581491367e+16
pdl = 0.138255
peta = 1000000000000000
phi = 1.618033988749895
pi = 3.141592653589793
pico = 1e-12
pint = 0.000473176473002
point = 0.00035277777777777776
psi = 6894.75729317
q_e = 1.602176634e-19
quart = 0.000946352946004
r_g = 8.314462618
radian = 57.29577951308232
roentgen = 0.000258
ry = 2.17987196968e-18
sigma = 5.67040047374e-08
sigma_t = 6.65245893699e-29
tbsp = 1.47867647813e-05
tera = 1000000000000
texpoint = 0.000351459803515
therm = 105506000
torr = 133.322368421
tsp = 4.92892159375e-06
uk_ton = 1016.0469088
us_ton = 907.18474
v_m = 0.022710981
week = 604800
yard = 0.9144
year = 31557600
yocto = 1e-24
yotta = 1000000000000000000000000
zepto = 1e-21
zetta = 1000000000000000000000
