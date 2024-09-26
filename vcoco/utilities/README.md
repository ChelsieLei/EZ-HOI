# Visualisation Utilities

## Dataset navigator

A command-line style dataset navigator is provided, which can be used to visualise any number of box pairs of the same interaction type in the same image. To use the navigator, refer to the example below.

```bash
user$ python navigator.py --partition trainval

***************************************
* Welcome to V-COCO Dataset Navigator *
***************************************

Commands are listed below:

path(p) - Print path of the current node
list(l) - List all navigable nodes
move(m) - Move to a navigable node
help(h) - Print help manual
exit(e) - Terminate the program

> p
/
> l
images		classes
> m classes
> l
[  0]                       hold obj	(3024)
[  1]                      sit instr	(1072)
[  2]                     ride instr	(445)
[  3]                       look obj	(3127)
[  4]                      hit instr	(342)
[  5]                        hit obj	(228)
[  6]                        eat obj	(403)
[  7]                      eat instr	(153)
[  8]                     jump instr	(489)
[  9]                      lay instr	(283)
[ 10]            talk_on_phone instr	(292)
[ 11]                      carry obj	(464)
[ 12]                      throw obj	(261)
[ 13]                      catch obj	(256)
[ 14]                      cut instr	(247)
[ 15]                        cut obj	(232)
[ 16]         work_on_computer instr	(398)
[ 17]                      ski instr	(437)
[ 18]                     surf instr	(485)
[ 19]               skateboard instr	(467)
[ 20]                    drink instr	(125)
[ 21]                       kick obj	(134)
[ 22]                       read obj	(86)
[ 23]                snowboard instr	(367)
> m 19
> l
[7] 1               [15] 1              [18] 1              [30] 1 
...
[4940] 1            [4954] 1            [4960] 1   
> m 15
```
![tmp13t3unw8](https://user-images.githubusercontent.com/11484831/104798299-c5269980-5819-11eb-8dab-031ec72debbc.png)

## Generate and visaulise box pairs in large scales

Apart from the single-image visualisations, to get a more holistic view, utilities for generating and visualising the complete collection of box pairs are also provided.

```bash
# Generate all box pairs from the training set
python visualise_and_cache.py --partition train
```

To visualise the box pairs, select a directory containing generated images

```bash
python generate_html_page.py --image-dir visualisations/train/class_000
```

This will create a file called _table.html_ in your current directory. Open the file to visualise box pairs in a web page. For more layout options, refer to the documentation for class [_pocket.utils.ImageHTMLTable_](https://github.com/fredzzhang/pocket/tree/master/pocket/utils) and make corresponding changes in _generate_html_page.py_
