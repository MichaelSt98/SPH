# Multi GPU

## TODO

* distinguish total number of particles and number of particles on (local) process
* sending particles between processes
* gathering particles (for rendering, ...)

## Send particles

* send particles
	* `x`, `y`, `z`, `vx`, `vy`, `vz`, `ax`, `ay`, `az`
* delete (sended) particles (from (local) process)

Example: 2 processes

* process 0: sending 100k and receiving 200k
	* 300k particles belonging to proc 0
	* 100k particles belonging to proc 1
* process 1: sending 200k particles and receiving 100k
	* 200k particles belonging to proc 0
	* 400k particles belonging to proc 1

 


## Reset arrays

* `resetArrays()` needs global lengths
* generalize resetting of `procCounter`


## Compute bounding box 

* `computeBoundingBox()` need to be *globalized* --> **reduction**
	* compute locally
	* share 
	* take global maxima 	


## Build tree

* `buildTree()` need local lengths

* need for inserting particles retroactively
* need for new flag
	* domainList: save indices (no flag needed)
	* toDelete: saving indices or rather index, from which on deletable? 
* need for function to delete particles/pseudoparticles from other process(es)

## Centre of mass

* `centreOfMass()` needs local lengths?!

## Sort Kernel

* ?

## Compute forces

* ?

## Update Kernel

* `updateKernel()` needs local lengths