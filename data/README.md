## Manual Download Links

Manually download the benchmarks from OneDrive and place them in **this directory** before running `./download_data.sh`:

- [OneDrive - iccad2015.tar.gz](https://1drv.ms/u/c/2c6e2bbdaffc31ad/IQDxRUC5lKljRIiyy6f227TUAfOnip74xQk7EYb65zygpEs?e=KIB8Hb)
- [OneDrive - iccad2019.tar.gz](https://1drv.ms/u/c/2c6e2bbdaffc31ad/IQC5ZBVuvgDbSbDfdv7CtgJpAbRkgirjE2Qp0X01xXIQ0GY?e=VaDLH1)
- [OneDrive - ispd2005.tar.gz](https://1drv.ms/u/c/2c6e2bbdaffc31ad/IQCreTapLGQ_TYmaPNbrDqZ4AY4pAo-vuAgit-Wf8i6hPW0?e=iOOy7M)
- [OneDrive - ispd2006.tar.gz](https://1drv.ms/u/c/2c6e2bbdaffc31ad/IQAvFi3DRJWsQZhJBaTqUG-GAXbe11jQy7cnyzYOrVVXdm4?e=Slxuki)
- [OneDrive - ispd2015.tar.gz](https://1drv.ms/u/c/2c6e2bbdaffc31ad/IQBey6FZ1ZJYQY-PafD4Uj0rAQeADcsDb9S0A0EjIcBno38?e=UCu1j5)
- [OneDrive - mms.tar.gz](https://1drv.ms/u/c/2c6e2bbdaffc31ad/IQAgtUqXnm1ES4YA6tB7oZrlAaG5HDgTT9SGR5XHXE9sYlQ?e=XHsI4R)

---

The following script will automatically extract/download benchmarks in `./data/raw`. It also preprocesses `ispd2015` benchmark to fix some errors when routing them by Innovus®.
```bash
./download_data.sh
```

# Note of Fixing ISPD 2015
We provide a python scirpt `fix_ispd2015_route.py` to fix some errors in `ispd2015` benchmark. Thus, Innovus now can detailedly routed them.

## Limitations
**removeDefSNetVias**: Due to numerous DRVs caused by SNet Vias (spacing) after nanoroute routing, we have enabled `removeDefSNetVias` to remove these vias and address the above issue temporarily. It is likely that these vias are oversized, directly violating the spacing rule. While this adjustment has no significant impact on placement, it does result in open SNets. We sincerely encourage and appreciate contributions towards resolving this issue. Your contribution is highly valued and appreciated.
