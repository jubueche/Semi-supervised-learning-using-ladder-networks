function [y1] = nn_extended(x1)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Generated by Neural Network Toolbox function genFunction, 29-May-2018 12:13:12.
%
% [y1] = myNeuralNetworkFunction(x1) takes these arguments:
%   x = Qx128 matrix, input #1
% and returns:
%   y = Qx10 matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0.4255;0.59425;0.77461;0.39503;0.71266;0.54242;0.44744;0.57207;0.64259;0.65918;0.57209;0.81725;0.67288;0.59136;0.44286;0.36654;0.33832;0.26069;0.5373;0.43061;0.69946;0.52934;0.55276;0.15958;0.44367;0.42227;0.35322;0.43192;0.57579;0.25125;0.29929;0.46597;0.6841;0.61251;0.75304;0.31464;0.56315;0.42443;0.63236;0.57523;0.44339;0.51248;0.31295;0.74379;0.67992;0.29882;0.66025;0.4348;0.55811;0.59711;0.29507;0.63093;0.64152;0.66689;0.56279;0.34328;0.38219;0.61311;0.63526;0.56447;0.57711;0.29949;0.31319;0.52055;0.42202;0.49103;0.56006;0.7609;0.50021;0.71011;0.78466;0.44868;0.28956;0.39685;0.59649;0.30135;0.73625;0.5082;0.41088;0.43565;0.47228;0.44719;0.61584;0.36108;0.62488;0.38369;0.51181;0.63733;0.62607;0.25817;0.40816;0.71607;0.36626;0.63744;0.5536;0.7306;0.4551;0.61517;0.67996;0.3966;0.42649;0.33412;0.54628;0.4292;0.55815;0.35508;0.41345;0.60374;0.6014;0.62575;0.68828;0.68727;0.4898;0.37064;0.4763;0.12243;0.36442;0.50582;0.50978;0.6233;0.47548;0.54963;0.58964;0.48317;0.31344;0.51331;0.54488;0.44969];
x1_step1.gain = [95.0570342205323;50.5561172901921;106.213489113117;86.6926744690072;95.4198473282439;89.9685110211427;73.3944954128441;75.301204819277;64.516129032258;72.2804481387785;70.1262272089762;110.741971207088;66.7556742323098;72.6744186046512;81.6659861167823;72.4900326205147;83.752093802345;95.0570342205323;76.9230769230768;69.7836706210746;111.919418019026;67.8426051560381;81.4995925020375;133.422281521014;61.8811881188118;82.6446280991736;76.1904761904762;45.1467268623025;78.0640124902421;95.5109837631327;105.652403592182;64.4953240890036;85.1788756388418;59.7014925373135;92.7643784786646;74.3770918557084;67.0915800067092;63.2511068943706;86.1326442721791;97.9911807937289;76.219512195122;82.9531314807135;73.1528895391369;77.9423226812159;108.108108108108;94.3841434638979;65.2528548123981;94.7867298578199;66.6666666666666;74.794315632012;78.9577575996841;78.462142016477;99.9500249875064;81.0044552450385;77.1010023130302;55.4323725055432;70.4225352112675;75.7288905717533;71.1743772241995;82.3045267489712;62.3441396508728;97.6085895558809;79.0826413602215;69.0131124913733;61.7665225447807;93.1098696461825;54.4069640914037;93.2400932400934;67.9347826086959;74.9063670411986;92.6784059314183;72.4375226367259;79.0201501382853;78.6163522012578;88.1834215167547;110.803324099723;92.0810313075503;71.8132854578096;71.8648939992814;77.760497667185;86.2812769628989;54.6597430992074;65.4878847413231;100.300902708124;86.8809730668982;90.497737556561;56.8504832291075;75.728890571753;76.3650248186333;71.9165767709457;58.3430571761961;103.626943005181;90.8677873693774;70.2987697715291;74.7384155455904;101.061141990905;80.9061488673138;70.3977472720872;101.936799184506;64.3293663557414;69.5410292072322;71.7875089734385;74.6547219111609;71.5307582260372;68.1431005110733;83.9983200335993;74.4878957169459;64.3915003219576;80.906148867314;90.3342366757003;84.4238075137189;84.3170320404723;57.3394495412844;86.2068965517241;76.1614623000761;221.238938053097;76.5696784073508;68.7994496044033;62.6370184779204;70.7213578500705;64.977257959714;76.1324704986676;68.7757909215956;77.3395204949729;70.3482237073514;60.3682463024449;95.8313368471493;67.3854447439352];
x1_step1.ymin = -1;

% Layer 1
b1 = [-1.4593656332144984145;-1.4540037370925986693;-1.4614295690056078314;-1.4376667295174951899;-1.3690807213898832373;1.3895585585526659678;1.4626705707705751625;-1.4494715005972123034;-1.4370938826864982918;-1.5089785592178912577;1.4536488494901815827;1.3922649176633778989;1.438086208887473294;-1.4571667355565762758;-1.4522092818101455247;1.4743342260204048699;-1.4107193794514636398;1.3822792087110640225;-1.4280716776071546992;-1.3183554267278296823;-1.4118955156867740719;1.3920374181740879926;1.4621767903511624365;-1.4544112033034486409;1.3686532548093399697;-1.4027171765992223484;1.3759441723447696226;1.4623060458324104172;1.3362105185500017246;1.3852088147760741599;1.4060609979489469001;-1.3960195081478079526;1.3908626995812480853;1.2860228015405172908;-1.4101707696409053749;-1.3153389063507798795;1.3838877299712306623;-1.406946378798105135;-1.3486985084535243473;-1.4435827729427772592;1.4161583325378370724;-1.3404572972202237757;-1.2983253707575483915;-1.3805334694747588564;-1.3101060895317278021;1.2912297705697670658;-1.3593492881102178238;1.3189294739027823677;-1.2928199799237110046;1.249445037031428285;1.3182563668747107055;1.3060350802935591918;1.2251865912756110255;-1.2970527056889002981;1.352752924660244549;-1.3124939232078565166;-1.3306775174100289316;1.304691246551982875;1.3392924254558917774;-1.2420217315524999524;1.2690685399561270064;-1.2466955196670663764;-1.2503273314925560378;-1.3272686367816934894;1.290061016472346056;-1.269305177281434327;1.2825644581490149854;-1.3417621912281518792;-1.2564492705655783134;-1.2660793168164650169;-1.2377031688418527189;-1.3246908079009114267;1.2053505720953232583;-1.3533062529750379266;1.2779740422770191621;-1.2780824113110733631;1.2251973639449136311;-1.2220574405384789785;-1.2273703364202059873;-1.2698199094634927686;1.2457367772136163087;1.2566918901237562611;1.256176980927926401;-1.2218917143787131341;1.2290907691156347514;-1.2262677448448877637;-1.2050299136912658859;1.101673768102048756;1.1695946636059069146;-1.1605033947965399665;-1.1942834409133762374;-1.1927847038313430517;-1.1909683920819971181;1.1722222577121144127;1.1305127513822914587;1.1560217011299640522;-1.1577787706208373386;-1.2340440131417214076;1.1837882033433793083;1.2025641271332270232;-1.0713395747961242943;1.1649866014373477796;-1.211776094807121229;1.1893202898203607543;-1.2165151296771734568;-1.1533429676702791067;-1.1343109358705627532;-1.1159267409559618933;1.1523048257719090692;-1.1767883025708592015;-1.1794287110293666387;-1.1492845298032199342;-1.1089015995275679494;1.1800789109893792261;-1.1475499854053032855;1.1380852109814647566;1.1413030762979310406;-1.1412234076854514964;1.105407338255193217;1.184820152646674396;-1.1411940920096041996;1.0835429691084008663;1.1003308075229107921;-1.1538231920793642349;-1.0978022577978789887;-1.1039514550937203907;1.0439655582184845795;1.0486705234284738886;-1.1043724726917849566;-1.1171638933004777172;-1.1210964032903503185;-1.1324721509956392484;-1.12715737080343259;1.0565339768501351703;1.0938307059400436128;1.0848790334412323766;1.0876934168295353533;0.99866361875992193387;-1.1216809274223369375;1.0671304949915201288;1.0932323635088259106;-1.0562922776050456264;1.0033034281275592381;1.0466018372088286004;1.0360635382461986786;1.0437708903224114376;1.0499439140305713014;1.01999711008282401;-1.0362686882623075579;-1.0767467540962118999;1.0455904867207186371;1.0496169526200065381;-0.91511936937015481153;-1.0484842205797542114;1.0119829959981045953;0.99685751756887341912;-1.0570158348603688747;1.0055350967680740659;-0.98523694501681591262;-0.97280527277236095163;-1.0747684482375685633;1.0026171923857194468;0.98168607373820071516;1.0473848482586454178;-1.0061929825905648883;-0.99089378071040234985;-1.0666459274292121595;-0.93780018660495578775;-0.99893883639491465321;-0.89397748463949699982;-0.97114175897893273426;1.026940118313404593;-0.97257019552448009669;-0.8740887060780292428;-1.0769622806013556815;1.0137362275650709798;-0.88787466624856825614;-1.0275337126060590798;0.95195973740344808078;-0.96323699607746060547;0.82971017332038665693;-0.90694723444881109042;0.97495816703096427336;1.0084709097106914211;0.94848790782716496661;-0.96606588464842468866;0.9155148412981434225;-0.91644761141357822254;0.85026584594286491736;0.90460543368487500437;0.76625754587859062905;-0.90792852278138957978;-0.95033278018981914848;-1.0098808545753130694;0.84034411395249974941;0.87094327437779694989;-0.99326218580519065604;0.96489786458564152927;0.9402262997118731036;0.93942057064681516731;0.90677336064387659142;0.93915749569802875474;0.92322653885881922342;-0.91163804090407463132;-0.80895178004846468589;0.88162631870947216761;-0.99974833850521371836;-0.89355144626702853738;-0.81602337422307680281;-0.92800131235580984868;0.88118983944399720265;0.86394989178895786885;0.87419207415605804101;-0.77380641505465608265;0.88021262626439633614;-0.82477192959817535733;-0.85893134399990067607;-0.85799733784785781054;0.86476933028231983691;0.80810238213958041076;0.74504353352862462412;-0.83385714271946254073;0.91972115924910424756;0.80295473090635505198;0.80331357781180923183;-0.83555812346447588812;0.92810472407278810092;0.78810349571711713068;0.80513970654773969748;-0.83359657606593462464;-0.72460846548560398084;-0.82061075361889124213;-0.77451819958736678462;0.789234573245402804;0.74070500611867351459;-0.85987708738008783449;-0.71529869568425874515;0.83834169017044468664;-0.85289974449439487358;0.70312580066665819611;0.73568694354316588324;0.75577340214514387995;-0.79554903676601851359;0.80052301664601543951;0.72736114487282199637;-0.69845224772372471733;-0.65729229859021942595;0.74242683484521410087;0.76200945932744290623;-0.76953298440814987824;0.75058546090866451461;0.89737489336471043533;-0.79535014586431007633;-0.81191485593305234847;0.66645818262631217888;0.7222296890674392289;-0.76748279091841853372;-0.72107076205359710297;-0.70433928167857928759;0.77014136330192228819;-0.78508391658845533279;0.80362315748691182105;0.71414196820932396292;0.70816518198939371054;0.76198914786513582964;-0.7182536214930774765;0.66829717665446231045;-0.64183673354768000507;0.81603442863541630903;-0.71247134022334435155;0.67610480906062775386;0.73082811547075010594;0.78031330442385926638;-0.70062571325997125005;-0.73704092042917168648;0.74479795922761360405;0.76704704327721140444;-0.72263720969546396677;0.65021411984219135149;0.71180914772524517264;-0.62415065002788661097;0.72907362983357104635;0.64050583099357183414;0.70008830728946447319;-0.67385791268957573319;0.69321165224899761181;0.61596892751478304895;-0.59976961374411297268;-0.61547392096763164915;0.53559718474717199488;0.79032418342525578847;0.66746326966470692632;0.41695477786588464042;-0.71869068355708898288;0.57518902724258857972;-0.70280981474506376561;-0.48992337759359683158;0.62495582980964148323;0.66028560602460339979;-0.60313818828851983866;0.67307919528074877658;0.59391322210290464767;-0.69909645712113765992;-0.62529082976068817157;-0.67039571454270929785;-0.61786988644227469614;-0.6299747366909184132;-0.53857226011016423595;-0.68579852049089162769;-0.61380254898642050421;-0.75971501370759941807;0.623882317732890046;0.50934155176112949626;0.45938617228007205817;-0.52948898135215238803;-0.64324372363721527002;0.62943608126260464086;-0.57496374086938528691;-0.58341250446048764644;0.56104650289576385447;0.64861125541756781132;-0.45344775939814868648;-0.52098758136751077696;0.53864700828179068459;-0.50278657887205013832;0.54510387370122992401;-0.46110785235502160795;0.4861547713114286462;0.72469266037616264509;0.51085889436638665106;-0.51491693034411478713;-0.56700083951046575947;0.50695363674925275621;0.53838537909243588953;0.47473052064103077319;-0.51611895138849683828;0.55869281415844362915;0.5348562699494580075;-0.42378739816712412969;0.55011764903068405275;-0.41228998395407379096;-0.64811853002842878535;-0.51368076563039988258;-0.40439683152117905651;-0.61922317187424069651;-0.48454883923492469977;0.45207250224319744936;0.47193247159275325542;-0.46909252129691231259;0.46402572050681489824;0.46462993914417188845;-0.435930966288370636;-0.41276891016860206562;-0.73824038043511353013;-0.52985893749514412221;0.44061491483867160079;-0.40619636186382668264;0.42119387793080464988;0.42658471084156412978;-0.46762551808163449474;-0.5266563601473871481;0.48842249530726072893;0.45763791203980136446;0.56346259884241955973;-0.61689156881368811813;-0.49247550308350834802;-0.51459965725605405495;-0.51700482000944392169;-0.47044852887828153554;-0.51514278530890666996;-0.46207895629825757045;-0.44978890024405476789;-0.35547129253479187172;-0.46556484946944648451;-0.37626474990129765708;0.42233451070655425585;-0.30594648037144162034;0.38194556480218305738;0.29123719941642706166;0.47612201548078320057;0.39840018736597387905;-0.41368691389540968029;0.46694083604339797766;-0.31977679744935388895;0.41068632473558575047;-0.39754244541939798285;0.33811706082007464413;0.31042866399549212675;0.36552157710377608524;-0.42117140020215571017;-0.48006243233204781706;0.26943488284806693667;0.37972415631711653461;-0.39959281650825506649;-0.44415224271907616238;0.32582338225278179022;-0.36053877872828327122;-0.2897568043583944597;0.4877609226457102487;-0.31805856738451659949;-0.34546395749070712977;0.46327816234760132996;0.25667720838178648135;-0.33570765880533065539;-0.28990488349694065739;0.42961796911343158589;0.32662921502952141362;0.25535281409752080828;0.25304528915567625624;0.28370542835177908758;0.4603538655126556689;0.13809854101990068354;0.36413569830595327037;0.26515667879958554343;0.32704954312616640877;-0.23322253103853540779;0.4022077630220380895;0.26823493666560266346;-0.1258620145110547206;-0.36100161677790432169;0.29084921224343590618;0.11521311777174618463;-0.29115267757637741664;-0.3572059710605417493;-0.19490569274973729152;0.2497048436467604049;-0.20818444979584263543;-0.22828728883534527005;-0.13470297349155987976;-0.1964464632692367041;-0.34752462397175903908;0.10373305104884381067;-0.26943528820342588226;-0.14103617251692895929;0.24996577645310727456;0.15421747534650467881;0.38180076891265074535;0.23935998003050251715;0.079057769855288798078;-0.19108185237571767567;-0.30730377722393459328;0.33123176108446394927;0.11078486173771542878;0.36806550335423737774;-0.051419706060397321612;0.1241674605251915503;-0.18528564387079765607;-0.20074190426133878273;-0.22293499682255735195;0.26195191057184835648;0.17909385868027782873;-0.19976396958407247051;-0.11670993390611797413;0.18424476455341570746;0.03858061756998969194;-0.14534454019446776951;0.32275500683493124621;-0.15048481786617440292;-0.074064031058794110862;-0.18824519001186129641;0.15086500835113050845;0.17440101089897924735;0.21095497099669960339;0.1653803438507753798;0.030182867140288387187;0.087738314112829354774;-0.15513031679798036655;0.010283772341040831044;0.10279031523824909422;-0.098191508408058986168;0.1830101482005472846;-0.067194782283129736444;-0.059180640358808524193;0.0041111000856000447581;0.18371934987227833691;0.27497935637088039007;-0.16441544020242696011;-0.055355237926667348602;-0.15781502076589448702;-0.30375257632792423967;-0.0051529244412410442691;-0.0078297188301762951634;0.11845517175902246787;0.016832134589930755619;-0.23425627315576294518;0.050332085573055039551;-0.013551616989939859956;0.046339512437059027228;0.12407980750039859785;0.070092933320863803903;-0.21006525972038989081;0.01586734355792116663;0.1044963133710277714;-0.087639997775099176547;0.085479164230314735873;-0.20938828588667937614;0.020526163468586950539;0.018678917426609632779;0.081508369353355786258;0.21867916062526873699;0.050055669839356789486;0.087985410549385051349;-0.058498284054099070062;0.0036096210449134912522;-0.077267092731053213517;0.091870095418581515512;0.061286637371097331395;-0.079444987166126745937;-0.0171877815543355697;0.098153951753779930534;0.006604830861191252761;0.11931070718118624852;-0.097353255332658053534;0.079247575963705213509;0.079688337193345101794;0.09347284275935034692;0.12037529903855656577;-0.06635710067830988268;-0.10625721687371025848;-0.069678495447984198363;0.027502780083119533067;-0.201776236980622109;-0.035313256675396961781;0.17498802598832677302;-0.059318113649742612581;0.15495295837839812014;0.0094879688813893671429;0.27901672030834828719;-0.14161548835464873863;-0.18214680403822036681;-0.14994062057544466082;0.027775056390400360701;-0.19023238628370309211;-0.18728479178151355433;-0.066395042714930449512;0.042141566347865137832;0.16166818601520702159;0.14509610911165643499;0.12456001027601192044;-0.23421254554616235954;-0.0078600757754062264282;0.22768950600459414435;-0.12597879398277497809;0.25739625254611692151;0.12300551225735625871;0.30864459834987356679;-0.34375343721892748228;-0.1658728310661113925;0.27251279439512654523;0.21278553293931529167;0.068000256524260171975;-0.20219185419012594696;0.1288403614624840654;0.23981578882038984135;0.02092319175097123729;0.18896670905374818306;-0.33004253812048639505;-0.23640986415249234942;-0.35145032119373520452;-0.31176224850816408596;0.20613051469552051698;0.13221801118908341199;-0.15311372821337307371;0.25667314745613373228;0.17692783453881028022;-0.16268775909436802851;0.21578516169003364444;-0.44508964106349496737;0.31252277118608301487;-0.050787856177765461352;-0.29743220016278537621;0.21186482779282225786;0.016796479701691627989;-0.062090244371282729552;-0.1551107022488837639;0.36352878375187036575;-0.30840762582937358838;0.25937669716506883688;0.2846773481886626378;-0.33004651153424141574;0.2112655749272907002;0.23291983928627674194;0.29835227239640382813;0.21509277354619704692;-0.47107861617226270967;-0.19196087694678251601;-0.32675282417862661077;-0.16388565801003759925;0.41215367608041908465;-0.2449063521159697554;0.39876707600996663672;-0.16294121376254633304;0.21238897976754469998;-0.41385043186676728766;0.30270191591975192935;0.36093937834303219114;-0.32855140753001160769;-0.24982925852609369488;-0.33828423362505988248;-0.17973027250941750466;-0.19918521198404007255;0.23314761270221892597;-0.30827924233032250045;0.18485712943299842381;-0.29983539843378664846;0.37506636104626600581;-0.41799977164854890122;-0.23354105108080822073;-0.4399771215420712478;-0.36002857253771008983;-0.50929965866246551531;-0.2094031507204424658;-0.24843820728881169102;0.35298445244621168282;0.26732189781250714766;0.27210957587757733656;-0.36090931392832964475;0.47117262509313645458;-0.4187151907738252965;-0.3988874712670545386;0.34412802019527177766;-0.28617808055090560027;0.44187748753861022522;0.34073487399326402025;-0.45802312333413136569;-0.17605459931008229923;-0.40015230939641038832;-0.3536857407757540428;0.53638750097488308022;-0.43268651503420474391;-0.32337391923195257792;-0.47942035708262070726;0.42236051401226903934;-0.43299723778729964918;0.4527245530466056378;0.48035389887058421721;0.46113928238707035945;-0.63856901394879928002;-0.27921090330039149974;0.44922442350610947148;-0.56054182352536907619;0.32406736337595415387;0.49325609265724856822;-0.48142885961657500005;0.51409870236056887105;0.35487943618942452417;-0.52220332062823060504;-0.46228284294250471254;0.3086828566864529555;0.5327974031675661859;0.48533369262171244252;0.64671052164580633548;0.39158054198819614022;-0.5702069313545006235;0.45788927982643012893;0.43565152657562233873;0.45437575487907022609;0.44590128724713895547;-0.57858707904835315894;-0.58346586477873052523;0.35261682705563707207;0.6303134421475141691;-0.54927010410984622446;-0.55240037183780010821;-0.52934764695334801665;-0.53969309361101669431;0.45986395317165512653;0.44625995343233987311;-0.64525675493037415453;0.60744766841494612919;-0.55192906660755014503;0.59907266345049847711;-0.48509377076556647879;0.45842071360833797078;0.31445706977067150456;0.51475388616217288007;0.512844769476356932;-0.50937023422774652115;-0.61022999932745092266;0.60788117107233641256;-0.46341780045083519335;-0.55416540336031250291;0.52266223260642508119;0.57237019420195356378;-0.52176040326549621984;-0.66754727835636107347;0.52455148712131305366;0.59097112018361419583;0.62026340136795810043;-0.6064723560040957695;0.65391372589274121552;0.57946075262253360361;0.61079614982294627712;0.54103549743103773739;-0.68892007928206178136;0.59715474063301232643;0.6309792956583150092;-0.68598525288485612261;0.72918176633715037216;-0.66351639293932973551;-0.59780323850156547039;-0.59196971940695886083;-0.63072131251805674257;-0.60177513031912777119;-0.62209209832946488206;-0.60816673794344766613;0.54776264956320730448;-0.68793564522675676454;0.71787218375776640222;0.68247943227002316746;-0.65588646374258363636;-0.71640489865438194972;0.58758852424488727806;-0.7158761550138608154;0.58241106696480182681;0.64829540387662776268;0.8622846283137360901;-0.70557297733159274333;0.69947110413090507475;0.60000956812830164022;0.67552687390010535129;-0.72816759814812370699;-0.74148937933667280475;0.64984332327027849452;0.74252435543038675103;-0.64609667887806643272;-0.63891281345690886795;-0.67475398938559505435;-0.6884726200810631358;0.65183067410072781378;0.69395833516784977135;0.70104395747436298869;0.78288813169609083875;-0.72772347455789598847;-0.78321667042233666933;-0.71715185941136005976;-0.71999497123670519461;0.66358293170279702977;-0.7163475448297023096;-0.67695558022394808351;0.72260788102682971257;-0.69222713835080551537;0.77320758770878705857;0.92033911538348089909;-0.71537222935596822548;0.67048068975322439034;0.72550872647132969018;-0.72495307971859201945;-0.80185189946576296283;-0.81141028254838032385;-0.70428979933886937115;-0.7421125192160678008;0.83833253502461413387;0.76552056965322756721;0.80860055311195377925;0.72066721450551196604;-0.7821262582609187497;0.75444606782072654028;0.78006121559611785177;0.7748928128809782212;-0.760233124060088028;0.81023298987652658809;0.77109347733598265773;-0.92602177479139491378;0.81494712246766198849;0.84653192050016445869;0.75650642108096044502;0.79004751981411702744;0.76622297263443961413;0.77869508224359740289;-0.83838557617504050334;-0.74635371716482279414;0.80747638255232079274;0.76989325385264617552;-0.81346555098403217077;-0.87817938072392065507;0.79028445395929369788;-0.81699864455352222414;-0.89573656569991699783;0.83191882894475110977;0.71425913545889752054;-0.87735938796309465015;-0.7902566082258928315;0.83866348520763001062;0.88951503942057463803;0.8671583903662707371;-0.81614180439404138134;0.916467582069882436;0.86513340032657182199;-0.96278698984791100113;0.83411888745771567422;0.89645557219246674308;0.94520661723485521755;0.80490956988081607815;-0.85699826416632507442;-0.8385569169338705775;-0.81906064898405561703;0.92392024425697383272;0.90345888185006539839;0.94710048725609752296;-0.93844477939435289038;0.88501720647791293217;0.90044971954385688573;-0.89935506722658964396;0.94260372950196336106;-0.98629771121817588764;0.90672581872057422814;-0.87443359228940620298;-0.96982579521137812772;-0.91067798770352514914;-0.98309006308647617356;-0.96273589677633164552;-0.99161146813838374481;-0.94453355902911539932;-0.91168846949875936847;-0.955440718432164382;1.0493191053688790682;0.90451358520074898184;0.92203419456349655636;-0.98087689678709288899;1.0447158492017143416;-0.88116708073914673527;1.0268067126653517285;-0.97310766237673462786;-1.0115543121768537649;-1.0392718082170873117;1.0466266537068689146;0.95228557890016762855;1.0806294166040188998;1.0797575387747551101;0.99834173963455641321;-1.0187982406903368648;-0.96276142914591700261;0.86900692597091688807;-0.93025717596487700334;-1.0036376218842075758;0.97652298198285913955;1.0140043891479235683;-0.98861542875936569974;1.0162492796984790022;1.0640607921494982246;-1.0545537263248283555;-0.97902810426033148872;1.017705911811169095;0.99823128205080136066;-1.059889335502066432;-0.97444737395116931555;1.0055719006070731325;-1.0470498760046391684;1.0918964792709864309;1.090860282807810222;-1.0152863383359791438;-0.97714233277503748099;-1.0764678421876825443;1.0034819346597856349;-0.92751768697690561982;-1.0447970286287415753;-1.0891151168961095141;-1.1415633473640629525;-1.1622722838505534781;-1.0992302486565903585;1.0663389722283385108;-1.0148185467919559333;-1.0264746931409853836;-1.0265730039094353998;1.1222221912574441571;-1.1079924686995090699;1.0746820726512229527;-1.076004121271018521;-1.0561564102187388503;1.1614926719003404454;-1.1253809706577186933;-1.0180730846107131082;1.0833561845462791329;-1.1308971373264833371;-1.1398710283703876733;-1.1805887907638390466;1.1203829146644486148;-1.0709217181836281352;1.1619587309595669211;-1.116879528317626491;1.1356347564444164711;1.1180216039553052543;-1.135762220492155361;-1.1488321354265094154;-1.142111909762159927;1.0630386391369144317;1.2144141249892372869;1.1991172713578415188;-1.1485062397000846968;-1.0904430194115872776;1.1152308495020510914;1.1084999167258280384;-1.1729694373654151462;1.0779161018374847636;-1.144032569312537051;1.1090099728034765736;1.153710739498713389;1.1431724797620195222;-1.1600489147815504776;-1.1012656457350411543;1.1649256473913394139;1.0867533074450039798;1.2621790576695952613;1.2082013035613545782;1.1441218217320168193;-1.127011082706833589;1.2510998632463323776;-1.2698818625181351738;-1.1670708954156652215;1.0924809986690482422;1.1287645946623674487;-1.0705293682375880415;-1.1726972267792665772;1.2027035787002904321;1.2753004614886009893;1.1354064479386285491;1.1691844387214676004;-1.3080222895268120986;1.1187101707249347271;1.2231458306827400762;1.2255661203724472763;1.2187792507340593495;-1.2483749208351633264;-1.1703494159087559634;1.3099851044616674933;-1.2508819523403762464;1.2878128750442698891;-1.2708069688298599953;1.284664029668717955;1.2540088731768805896;1.2084922697325337637;-1.233794278230604613;-1.2047896390559436064;-1.2843764609129397769;1.2504912166306172416;-1.3445569899785596579;1.2671453733973461198;1.2926859617409052206;1.1201152828940776907;-1.2429210657304661236;1.2660782224582300959;1.3099738261074280743;-1.2842643445426904503;1.339522102228868139;1.2910653514390950658;-1.3450276660036997178;1.3410610725558889111;-1.2674842455333226177;-1.2769718221891535048;-1.3914823404527454365;-1.2850381268869477402;-1.3777363448877721908;-1.3128838888932736761;1.3988093038277709113;-1.3879282313681573502;-1.4609965761579399857;1.3222123762129318614;1.3690505891934365845;-1.3549274192365861058;1.3158955670878897948;-1.4029020174901520868;-1.4078612352192778623;-1.4046126469555335614;-1.4644170771700095735;-1.3498994263059365117;1.4013442833775346941;1.4160358913787465251;-1.542149388710086777;-1.3959471008709363193;1.3845879060945600614;1.3855177650594829863;-1.4435872245660190671;1.3915355254607157942;-1.3591655043133386016;-1.4073094047816845364;-1.4134512870563162856;1.4039185992906872968;-1.4546274492786974708;1.3496664777239217869;1.3778935896372841441;1.398187268136414696;-1.3992979186812646297;1.4629155974922469774;-1.397302529851611741;-1.4113737422062520022;1.4586723833788957094;-1.4434002157168290825;1.5311253953312797815;1.3764146737825881939;1.3768830437904373554;1.3769800086001666717;-1.4883390358621728655;-1.3723839491730580598;1.4706772215261723069;-1.4741608926540332725;-1.4667037492118732978;-1.4417232013806022817;-1.473875091947712157;1.4541598119674565837];

% Layer 2
b2 = [1.0046989790966769363;-0.23321057942730266666;-0.61937421673660819632;0.48300114825495993331;0.75176089310580473946;-0.30596187135492852738;-0.68165453171816126066;0.97239023101738264465;0.14519158019818806382;-0.86282290394579042037];

% ===== SIMULATION ========

% Dimensions
Q = size(x1,1); % samples

% Input 1
x1 = x1';
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = softmax_apply(repmat(b2,1,Q) + LW2_1*a1);

% Output 1
y1 = a2;
y1 = y1';
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Competitive Soft Transfer Function
function a = softmax_apply(n,~)
if isa(n,'gpuArray')
    a = iSoftmaxApplyGPU(n);
else
    a = iSoftmaxApplyCPU(n);
end
end
function a = iSoftmaxApplyCPU(n)
nmax = max(n,[],1);
n = bsxfun(@minus,n,nmax);
numerator = exp(n);
denominator = sum(numerator,1);
denominator(denominator == 0) = 1;
a = bsxfun(@rdivide,numerator,denominator);
end
function a = iSoftmaxApplyGPU(n)
nmax = max(n,[],1);
numerator = arrayfun(@iSoftmaxApplyGPUHelper1,n,nmax);
denominator = sum(numerator,1);
a = arrayfun(@iSoftmaxApplyGPUHelper2,numerator,denominator);
end
function numerator = iSoftmaxApplyGPUHelper1(n,nmax)
numerator = exp(n - nmax);
end
function a = iSoftmaxApplyGPUHelper2(numerator,denominator)
if (denominator == 0)
    a = numerator;
else
    a = numerator ./ denominator;
end
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end