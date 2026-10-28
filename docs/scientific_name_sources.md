# 模型標籤的學名對照依據

本表對應 `api/result_mapper.py` 的 `PLANT_METADATA`。`scientific_name` 是模型**標籤所能支持的分類名稱**，不代表已對單張照片完成病原分離、顯微鏡或 DNA 鑑定。`spp.` 表示屬內物種未定；以 `-idae` 結尾者為科，`Pucciniales` 為目。四份 `中英對照表.txt` 只提供中英名稱，沒有病原或害蟲學名，因此下表另列可核對的來源。

| 植物 | 模型標籤 | scientific_name | 精度與依據 |
| --- | --- | --- | --- |
| 豆科 | Bemisia_tabaci | Bemisia tabaci | 對外中文名稱為銀葉粉蝨；[農業部銀葉粉蝨資料](https://kmweb.moa.gov.tw/subject/subject.php?id=19992)列有 *Bemisia tabaci*（*Bemisia argentifolii*）。影像分類不能判定品系。 |
| 豆科 | Scale_insects | Coccoidea | 總科；[臺東區農業改良場介殼蟲分類資料](https://www.ttdares.gov.tw/upload/ttdares/files/web_structure/5887/54-014.pdf)。標籤未限定單一物種。 |
| 豆科 | Small_green_leafhopper | Cicadellidae | 科；[農業部小綠葉蟬資料](https://kmweb.moa.gov.tw/theme_data.php?id=290&theme=pest_map)。僅由此標籤不指定物種。 |
| 設施小葉菜 | White_rust | Albugo spp. | 屬內物種未定；[農藥所蔬菜白銹病資料](https://public.acri.gov.tw/pdf/2022/%E6%A0%A1%E5%9C%92%E5%8D%88%E9%A4%90%E5%B8%B8%E7%94%A8%E4%BD%9C%E7%89%A9%E5%AE%89%E5%85%A8%E7%94%9F%E7%94%A2%E6%89%8B%E5%86%8A.pdf)列有 *Albugo* spp.；不同小葉菜的病原可能不同。 |
| 釋迦 | Algal_leaf_spot | Cephaleuros spp. | 屬內物種未定；[植物病理研究指出藻斑病可由多種 Cephaleuros 引起](https://apsjournals.apsnet.org/doi/10.1094/PDIS-01-15-0029-FE)。 |
| 釋迦 | Anthracnose | Colletotrichum spp. | 屬內物種未定；[臺東區農業改良場番荔枝資料](https://kmweb.moa.gov.tw/redirect_files.php?id=236986&theme=knowledgebase)記載炭疽菌，僅由模型類別無法確認個別菌種。 |
| 釋迦 | Kanzawa_spider_mite | Tetranychus kanzawai | 物種；[農藥所番荔枝葉蟎資料](https://otserv2.acri.gov.tw/PPM/PLC0101.aspx?ASParam=bUZ0SyUyNjZuJTFmJTE4JTAycCUxYSUxN3BKWVRkLm5jJTdmZXZxayUwMCUwMSUwYyUxMA%3D%3D)。 |
| 釋迦 | Mealybug | Pseudococcidae | 科；[農業部粉介殼蟲科資料](https://kmweb.moa.gov.tw/theme_data.php?id=211&theme=pest_map)。釋迦有多種粉介殼蟲，不能將類別直接寫成單一物種。 |
| 釋迦 | Rust | Pucciniales | 目；[USDA 銹菌分類資料](https://research.fs.usda.gov/treesearch/56741)。[臺東區農業改良場](https://www.ttdares.gov.tw/upload/ttdares/files/web_structure/8029/108-108-04.pdf)曾鑑定鳳梨釋迦銹病菌，但本模型的 `Rust` 標籤未限定菌種。 |
| 釋迦 | Sugar_apple_whitefly | Aleyrodidae | 科；[農業部粉蝨分類資料](https://kmweb.moa.gov.tw/theme_data.php?id=254&theme=pest_map)。番荔枝上有多種粉蝨，不能由此標籤指定物種。 |
| 釋迦 | Tea_red_spider_mite | Oligonychus coffeae | 物種；[臺中區農業改良場葉蟎資料](https://www.tcdares.gov.tw/ws.php?id=1743&print=Y)。 |
| 茶葉 | Agromyzidae | Agromyzidae | 科；[農業部潛蠅科資料](https://kmweb.moa.gov.tw/theme_data.php?id=101&theme=pest_map)。 |
| 茶葉 | Caloptilia_theivora | Caloptilia theivora | 物種；[茶園害蟲調查](https://kmweb.moa.gov.tw/redirect_files.php?id=562959&theme=knowledgebase)。 |
| 茶葉 | Oligonychus_coffeae | Oligonychus coffeae | 物種；[農藥所茶樹葉蟎資料](https://otserv2.acri.gov.tw/ppm/PLC0101.aspx?ASParam=JTdiREtaJTBmJTNjbiU3ZmRzdSUxY3IlMDMlMDglMDc%3D)。 |
| 茶葉 | Tortrix | Tortricidae | 科；[農藥所茶捲葉蛾類資料](https://www.acri.gov.tw/Uploads/Item/ce823959-11d0-4891-8142-2feb5e7db9c0.pdf)包含多屬多種。資料集的 `Tortrix` 為通稱標籤，不能推定為 Tortrix 屬或某個茶捲葉蛾物種。 |
| 豆科 | Liriomyza_spp | Liriomyza spp. | 屬內物種未定；[農業部斑潛蠅資料](https://kmweb.moa.gov.tw/theme_data.php?id=101&theme=pest_map)。 |
| 豆科 | Rust | Pucciniales | 目；豆科包含多種作物，不能將[菜豆銹病的 *Uromyces appendiculatus*](https://otserv2.acri.gov.tw/ppm/PLC0101.aspx?ASParam=TE9UTyU3Y2J1JTBjJTI2LjBmJTVlJTBlJTA4JTA2JTE0JTFkdyUxMWIlN2Vn)套用到整個豆科模型；[USDA 銹菌分類資料](https://research.fs.usda.gov/treesearch/56741)。 |
| 豆科 | bean_aphid | Aphididae | 科；模型英文標籤雖為 `bean_aphid`，原始資料夾和中英對照表只標「蚜蟲」。[農業部蚜蟲科資料](https://kmweb.moa.gov.tw/theme_data.php?id=176&theme=pest_map)足以支持科名，但不能據此確認所有圖片都是[豆蚜 *Aphis craccivora*](https://kmweb.moa.gov.tw/techcd/%E4%BD%9C%E7%89%A9%E7%97%85%E8%9F%B2%E5%AE%B3%E8%88%87%E8%82%A5%E5%9F%B9%E7%AE%A1%E7%90%86%E6%8A%80%E8%A1%93%E8%B3%87%E6%96%99/%E8%94%AC%E8%8F%9C/%E6%9E%9C%E8%8F%9C%E9%A1%9E/%E9%95%B7%E8%B1%87%E8%B1%86/%E8%9F%B2%E5%AE%B3/%E9%95%B7%E8%B1%87%E8%B1%86-%E8%B1%86%E8%9A%9C.htm)。 |
| 設施小葉菜 | Aphids | Aphididae | 科；[農業部蚜蟲資料](https://kmweb.moa.gov.tw/theme_data.php?id=176&theme=pest_map)及[十字花科蚜蟲多物種資料](https://otserv2.acri.gov.tw/PPM/PLC0101.aspx?ASParam=JTdiREtaJTBmJTNjbiU3ZmRwcSUxOHIlMDMlMGYlMGY%3D)。 |
| 設施小葉菜 | Phaedon_brassicae_Baly | Phaedon brassicae | 物種；`Baly` 是命名者，不是學名的一部分；[桃園區農業改良場資料](https://kmweb.moa.gov.tw/knowledgebase.php?id=415809)。 |
| 設施小葉菜 | Pieris_rapae | Pieris rapae | 物種；[農業部紋白蝶資料](https://kmweb.moa.gov.tw/subject/subject.php?id=13452)。 |
| 設施小葉菜 | Striped_flea_beetle | Phyllotreta striolata | 物種；[農業部黃條葉蚤資料](https://kmweb.moa.gov.tw/subject/subject.php?id=13454)。 |

若後續取得每張照片的病原鑑定或資料集採集紀錄，可再將屬、科、目層級細化為物種。現有影像分類標籤與中英對照表本身不足以做此細化。
