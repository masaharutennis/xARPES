.PHONY: all clean

all: xarpes.png xarpes_small.png xarpes_black.png xarpes_white.png

xarpes.png: xarpes.svg
	inkscape -w 932 -o $@ $<

xarpes_small.svg: xarpes.svg
	cp $< $@
	sed -i 's/23.3/7.3/' $@
	sed -i 's/466/146/' $@
	sed -i '/M\(7\|11\|15\|19\)/d' $@

xarpes_small.png: xarpes_small.svg
	inkscape -w 292 -o $@ $<

xarpes_black.png: xarpes.svg
	inkscape -w 932 -b black -o $@ $<

xarpes_white.png: xarpes.svg
	inkscape -w 932 -b white -o $@ $<

clean:
	rm -f *.png *_small.svg
