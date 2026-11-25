#!/usr/bin/env python3
"""Rebuild browser-native SVGs from the final paper's vector figure sources.

Usage: python scripts/build_paper_figures.py --source /path/to/paper/figures
Requires PyMuPDF. Geometry, glyph outlines and original observation crops are
preserved. CSS variables expose each vector color without changing measurements.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import fitz

FIGURES = {
    'overview': 'last-update.pdf',
    'retention': 'plots/F10_ood_retention.pdf',
    'rounds': 'plots/F03_multiround_ood.pdf',
    'ablations': 'plots/F09_ablation_gains.pdf',
    'training': 'plots/F04_vistamix_local_training.pdf',
    'fidelity': 'plots/F05_task_roi_fidelity.pdf',
    'cameras': 'plots/F06_robotwin_camera_coverage.pdf',
    'replica': 'qualitative/Q01_replica8_go2_g1.pdf',
    'robosplat': 'qualitative/Q02_robosplat_visual_worlds.pdf',
    'robotwin': 'qualitative/Q08_robotwin_camera_ladder.pdf',
    'piper': 'qualitative/Q04_piper_real_3dgs_sequence.pdf',
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    dest=Path(__file__).resolve().parents[1]/'docs/assets/figures'
    dest.mkdir(exist_ok=True)
    records=[]
    index=dest.parents[1]/'index.html'
    html=index.read_text() if index.exists() else None
    for name,source in FIGURES.items():
        pdf=args.source/source
        doc=fitz.open(pdf);page=doc[0]
        svg=page.get_svg_image(text_as_path=True)
        # Each figure has its own IDs, so all SVGs can coexist in the DOM.
        ids=re.findall(r'\bid="([^"]+)"',svg)
        for old in sorted(set(ids),key=len,reverse=True):
            new=name+'-'+old
            svg=svg.replace('id="'+old+'"','id="'+new+'"').replace('href="#'+old+'"','href="#'+new+'"').replace('url(#'+old+')','url(#'+new+')')
        svg=svg.replace('<svg ','<svg class="paper-svg" role="img" aria-label="'+name+' figure from the FedVista paper" style="fill:var(--fig-000000,#000000)" ',1)
        if name=='overview':
            # Original RGB glyphs sit on an embedded light gradient patch.
            def patch_ink(m):
                tag=m[0]
                transform=re.search(r'transform="matrix\(([^)]+)\)"',tag)
                if transform:
                    nums=[float(x) for x in transform[1].split(',')]
                    if 345 < nums[-2] < 380 and 233 < nums[-1] < 234:
                        return tag.replace('<use ','<use data-paper-knockout="true" ',1)
                    if 240 < nums[-2] < 295 and 370 < nums[-1] < 390:
                        return tag.replace('<use ','<use data-fixed-ink="true" ',1)
                return tag
            svg=re.sub(r'<use\b[^>]+>',patch_ink,svg)
        svg=re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{6})"',lambda m:m[1]+'="var(--fig-'+m[2][1:].lower()+','+m[2]+')"',svg)
        # The page canvas is explicit; bitmap observation patches stay untouched.
        end=svg.index('</defs>')+len('</defs>')
        svg=svg[:end]+'<rect width="100%" height="100%" fill="var(--fig-ffffff,#ffffff)"/>'+svg[end:]
        (dest/(name+'.svg')).write_text(svg)
        if html:
            if name=='overview':
                inline=svg.replace('class="paper-svg"','id="method-svg" class="paper-svg"',1)
                pattern=r'(<div id="overview-canvas">)<svg\b.*?</svg>'
            else:
                inline=svg
                pattern=r'(<div class="figure-view" data-figure="'+re.escape(name)+r'">)<svg\b.*?</svg>'
            html=re.sub(pattern,lambda m:m[1]+inline.strip(),html,flags=re.S)
        if name=='overview':
            previews=dest.parent/'images'
            (previews/'overview.svg').write_text(svg)
            page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(str(previews/'overview.png'))
        records.append({'name':name,'source':source,'source_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'viewbox':list(page.rect),'vector_drawings':len(page.get_drawings()),'observation_images':len(page.get_images()),'glyphs':svg.count('data-text=')})
    (dest/'sources.json').write_text(json.dumps(records,indent=2)+'\n')
    if html:
        index.write_text(html)
    print('Rebuilt',len(records),'native SVG figures from paper sources.')

if __name__=='__main__':main()
