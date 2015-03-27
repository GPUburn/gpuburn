Massively parallel processors provide high computing performance by increasing the number of concurrent execution units. Moreover, the transistor technology evolves to higher density, higher frequency and lower voltage. The combination of these factors increases significantly the probability of hardware failures, potentially resulting in erroneous computation.

GPUBurn proposes a set of functions to benchmark and locate hardware failures of NVidia GPUs. Based on this informations of corrupted units, it offers the necessary CUDA functions to quarantine the defective hardware and ensure correct execution.

This software approach allows CUDA users to continue using their CUDA capable GPU's even though their are known to be corrupted without paying the execution price of technics such as triplication, ...

---

## Related publication: ##
  * Defour, D.; Petit, E., "GPUburn: A system to test and mitigate GPU hardware failures," Embedded Computer Systems: Architectures, Modeling, and Simulation (SAMOS XIII), 2013 International Conference on , vol., no., pp.263,270, 15-18 July 2013 [link1](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6621133&isnumber=6621089)  [link2](http://hal.archives-ouvertes.fr/hal-00827588)
  * Defour D., Petit E., "Températures, erreurs matérielles et GPU",  COMPAS'2013 - Conférence d'informatique en Parallélisme, Architecture et Système, France (2013) [link1](http://hal.archives-ouvertes.fr/hal-00785386)