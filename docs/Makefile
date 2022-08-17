SPHINXBUILD = sphinx-build
BUILDDIR = build
SPHINXOPTS =
ALLSPHINXOPTS = $(SPHINXOPTS) .

.PHONY: clean html linkcheck docs

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf api

html:
	SPHINX_GALLERY_PLOT=False $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

linkcheck:
	SPHINX_GALLERY_PLOT=False $(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

docs: html linkcheck

