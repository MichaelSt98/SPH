# TODO

## Bounding box

* **Currently:** Bounding box or domain in dependence of particles
	* *no particle can leave simulation domain*

* **Fixed simulation domain**
	* How to delete particles
		* shorten arrays (how to efficiently?)

## Communication

* Sending arrays independently for 
	* x
	* y
	* z
	* v_x 
	* ...
* **or** creating struct class?
	* possible with CUDA (regarding cuda-aware MPI)


## Domain list nodes

* childIndex > n --> cell / pseudoparticle/ domain list node **mark as domain list**
* childIndex < n 
	* childIndex == -1 --> leaf **insert pseudoparticle (not! particle) and mark as domain list**
	* else --> pseudoparticle/ domain list node with leaf as child **insert pseudoparticle inbetween and mark as domain list** 	


* `buildDomainTreeKernel()`


Old ...

_____
* How to?

```cpp
void createDomainList(TreeNode *t, int level, keytype k, SubDomainKeyTree *s) {
    t->node = domainList;
    keytype hilbert = Lebesgue2Hilbert(k, level); // & (KEY_MAX << DIM*(maxlevel-level));
    int p1 = key2proc(hilbert, s);
    //int p2 = key2proc(k | ~(~0L << DIM*(maxlevel-level)), s);
    //int p2 = key2proc((k | ((keytype)iMaxSon << (DIM*(maxlevel-level-1)))), s);
    //Logger(DEBUG) << "p1       k = " << k << ", level = " << level;
    //Logger(DEBUG) << "p1 hilbert = " << hilbert << ", proc = " << p1;

    //int iMaxSon = maxHilbertSon(t, level, k, s);
    int p2 = key2proc(hilbert | (KEY_MAX >> (DIM*level+1)), s); // always shift the root placeholder bit to 0
    //Logger(DEBUG) << "p2 hilbert = " << (hilbert | (KEY_MAX >> (DIM*level+1))) << ", proc = " << p2; // fill with ones

    if (p1 != p2) {
        for (int i = 0; i < POWDIM; i++) {
            if (t->son[i] == NULL) {
                t->son[i] = (TreeNode *) calloc(1, sizeof(TreeNode));
            } else if (isLeaf(t->son[i]) && t->son[i]->node == particle){
                //Logger(ERROR) << "Deleting particle in createDomainList(): " << k;
                t->son[i]->node = domainList; // need to be set before inserting into the tree
                insertTree(&t->son[i]->p, t);
                continue; // skip recursive call of createDomainList()
            }
            createDomainList(t->son[i], level + 1,  (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), s);
        }
    }
}

```     	

