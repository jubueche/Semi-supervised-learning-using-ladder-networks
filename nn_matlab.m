function [Y,Xf,Af] = nn_matlab(X,~,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Generated by Neural Network Toolbox function genFunction, 29-May-2018 11:50:21.
%
% [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
%
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = Qx128 matrix, input #1 at timestep ts.
%
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = Qx10 matrix, output #1 at timestep ts.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0.42600888;0.5974661;0.77562934;0.39503255;0.71368;0.542416;0.4488222;0.572069;0.64258784;0.65918094;0.5732118;0.8181148;0.6736108;0.5927028;0.44471258;0.3665357;0.33913898;0.26087895;0.5379525;0.43171805;0.7001347;0.5293358;0.55400044;0.1595802;0.4464555;0.4252953;0.35562062;0.43191588;0.5776099;0.25125128;0.30051664;0.46659276;0.6864776;0.6143837;0.75303686;0.3146413;0.5633988;0.42657572;0.6328767;0.57523334;0.4458238;0.51416475;0.31350955;0.74500394;0.68107706;0.2988224;0.6609202;0.43635058;0.5581089;0.59755707;0.296104;0.63331586;0.6415199;0.667756;0.56354094;0.34450874;0.38274184;0.61310565;0.63525915;0.5664628;0.577112;0.3002342;0.31319264;0.52100986;0.424801;0.4910942;0.5635721;0.76090235;0.5014517;0.7111201;0.78505224;0.44867867;0.2902026;0.39874712;0.59826845;0.30234125;0.7374563;0.5088866;0.41261;0.43695948;0.4733847;0.44799137;0.6158364;0.3620536;0.62685597;0.38459474;0.5150839;0.637438;0.62708753;0.25928837;0.4110281;0.7160716;0.36626244;0.64079285;0.55360234;0.7309434;0.45567042;0.615168;0.67996055;0.39664367;0.42682114;0.33471268;0.5491165;0.4299316;0.55944455;0.35508427;0.41513097;0.6037396;0.6020175;0.6264156;0.6909222;0.68754137;0.4910546;0.37063542;0.47673452;0.12270298;0.36514196;0.5079669;0.5108966;0.62329894;0.47701097;0.55141795;0.59258115;0.48366073;0.31344447;0.51596457;0.5448817;0.4513466];
x1_step1.gain = [97.4344534073316;60.9575826661017;118.849536486807;89.8688588676971;108.257913653488;93.8671873166656;83.6172836925392;75.2893841591889;68.4840976501052;74.2179469386212;77.2272334115904;116.309229021014;69.9528168250513;78.5047668094404;98.8854619582684;80.6188301401558;95.8923550778838;95.9198110379721;79.6859100170767;72.5799486279125;123.486518359358;67.8364842249648;87.7363837519239;144.332113967524;67.7162000512611;96.9916111955476;92.9114163973784;48.3685645927112;86.0937130066075;95.5238936303235;112.973542726029;67.1679635680965;94.7584370543391;64.7900236321611;94.5414597390469;77.0485178369245;73.4156175574917;67.8500161822288;89.0571078704218;104.936969609204;86.124261807421;93.2429187830865;78.1038570418243;83.2221623947343;115.332272276429;94.394389197506;68.733010059076;103.135207162534;70.0433393162021;80.6213324851969;82.3087946123954;86.5480900134753;102.687643698521;85.5479560454603;83.1768222794109;57.3798194027564;73.4259136202866;76.8108149627467;74.1857648645088;89.637621985599;63.3954608850006;109.479681392231;79.0876449281094;70.6955094919327;69.6387801653434;96.1210356080375;61.2026941425963;93.2598756379555;71.0738620896993;81.987172287024;94.4027656234216;72.711593463664;85.2148052202588;84.9626695270766;97.9268398372259;117.435329831803;101.052971967905;77.5398166958731;78.5863262935702;81.9379303789793;95.5793590639339;60.1270846060233;70.1899691515084;111.880726195417;98.5934170161393;94.3697586446237;62.6835452559527;79.7273325227718;85.9933217586319;74.9217723044926;64.933547007992;109.313092921048;94.6727635926422;81.1473587346205;74.7332210840352;102.825278772184;84.7593385720255;70.3869947357567;107.203254690813;65.4828030699648;73.863371966339;74.179078681007;83.5034716568341;75.0132961067349;83.9622807849803;84.0095048353771;81.3805725855707;67.045788250628;84.3645493414503;94.0534693973523;98.0260974879346;91.8040552605333;62.3995756828853;86.1816364169122;84.4695187518107;237.519818059819;82.5362055382618;74.771833749314;64.9118821200222;71.796116260887;75.1166091461233;88.6204673400345;80.6734620612878;80.0883855422843;70.3603752881696;65.6361140401226;96.8211205592391;73.3484675671414];
x1_step1.ymin = -1;

% Layer 1
b1 = [-1.4348738327146020044;1.486499862008678452;1.4446022311995834198;-1.4442103074974426136;1.4910274139530630411;-1.465167301382153342;1.5108499085295983289;-1.4705481191794276263;1.4538390543646702024;1.445963436539091207;-1.4502797351050222829;1.4782493567695524916;-1.4163512713776054142;1.4338540799217103228;1.4613859737425209939;1.4636860476767299399;-1.4572910024622607317;-1.431346463996062468;1.3647903079067558174;-1.3887934316567149828;1.4143853822953977861;-1.4221279089662277606;-1.4055785202551942525;-1.3898439303225147423;-1.4110060875455943119;-1.3908594558394453689;-1.349698529386358814;-1.3836914756274480975;-1.4013746914522149378;1.3829818817730394898;-1.3906083644025735158;-1.3539879377821122386;-1.415267734248179643;-1.3754491612976893489;-1.4032533642525717443;-1.3681800441330953522;-1.388672098370801411;-1.3803069895321593563;-1.361362942639080531;1.3701009754213255842;1.35706788769070541;-1.3094534296521724848;-1.3490375745032050503;1.3234014879308348966;1.3481088418750397651;1.3716544004309534355;1.3374690041651307038;-1.3633094627216792993;-1.3637618997316909297;-1.3400865034741313853;-1.3460957716713899401;-1.3160697786449584434;1.2869396252609712761;1.3162640013571631492;1.3212871821348459722;-1.3485912678225815409;1.302257753832740006;-1.311090316272694567;-1.3236575296761965426;-1.2700254545321587596;-1.2949385998863287561;1.2746400792374918964;1.2572939905979070296;-1.2849920994033083854;1.2757763072726162612;-1.3070550087858909816;-1.2766581576207662874;1.3178198927168938148;1.2818027224282355636;-1.2932217350493306895;-1.2712295517989717908;1.2552697786798363566;1.2446302065423258121;-1.2057668247058941358;-1.2388754757608537727;-1.1947963240831516174;1.2678194894974310625;-1.2821555147751912962;1.2580384812484606272;-1.2299115770369570821;1.2294677382608185656;-1.2316530941611660843;-1.253189247234912207;-1.251432604952292138;-1.2117465331686063656;1.2353544725731691045;-1.2532590511513832521;1.2250333648419036159;-1.2571398960745865381;1.2218426191964626426;-1.1737216259163356646;-1.2187278761664586124;-1.1980130682558471911;1.1860847874676441194;-1.2222243296283048508;-1.2128804700123081428;1.1664149228566325078;1.1624368267783080633;1.1946197846133475018;1.2034440810405631073;1.1750851657330030342;-1.1884859609528755087;1.1731475222238310074;-1.1729108842045001815;-1.1749801219680702236;1.1725259406753236746;-1.1238991749695954603;1.1577721151936317678;-1.1646594442109217127;1.1335785735709655864;-1.1152281656026727941;1.1724693579732998749;1.1038936745004204187;1.1660106973927304441;1.1518436830220468714;1.1221005897459939149;1.1261388223331350034;1.1231976535920211724;1.12432025062916785;1.1533229665313444734;-1.1147800251157837792;-1.1338410475387961096;1.1245505122666219489;-1.0945992699057924824;-1.1010769126336661383;1.0851556324016735644;1.0837826013397537928;1.1148259129780637711;1.0935188111888114459;1.1217120554335122673;1.1092203652354173382;1.101591863161146545;-1.0952711913182988646;-1.0707804339679616312;1.124112551879344668;1.0885013710949116295;1.0481255729849801028;1.1045592495911180642;1.1078730148953432888;1.1059249394680976675;1.0663815162362564326;1.074000430028848152;1.0334073711970672704;1.0844932739920070119;1.0207109126121007847;-1.1185915703925244369;0.99818861721021912015;0.98415410279368764623;1.0367162915080236907;1.0315854274082840725;1.0491700795011345004;1.0522642361878971062;-1.0621872116280719478;-1.0423983384869401903;1.0112993886608470095;-1.0429282837912878801;1.0176499020731570866;-1.0065880731274785553;1.0035095707530294895;1.0083792716600568085;0.98615337373331302118;-1.0120889601189138052;-0.96516397987934832159;0.97262982915211215751;-0.97708860641833228389;-0.99389824668576409739;-1.003900185832712344;0.96382926453622230767;-0.99570699181405519784;-0.97571995147329881881;-0.99431354107686020782;0.93603349515503897482;-0.94794328776991676477;-0.97403417866665509273;-0.95035329955483149877;0.98538694634485202162;0.97231081927754947625;-0.97304275217445224122;0.90771462450623674467;0.94811387805789171246;0.96273414964458403631;-0.92898044042221283867;-0.92754901806227285643;0.95401762507056742102;0.89702127154758293237;0.93248401631301136572;-0.92472673917032310875;0.94048954933486073049;-0.88680072255788122693;-0.91361025786267491267;0.90352925630365232834;-0.89747481551235086528;0.92868916634587117542;0.95686102625531710419;0.96103641162871589199;0.96674582491035276899;0.94562899980738579497;0.89554818629032961752;0.92343365820329526006;0.89082498908077001332;0.89378075505720233629;0.8909658552610932869;0.89070850827851111298;-0.87295828238707962043;0.8871735526156417917;0.87522411692145052342;0.86087801522930862408;0.86198887731259077327;0.82210704011299207128;0.83555646184699616796;-0.86672405611609526499;-0.89467846597307554291;-0.83875078184021123473;0.84550225379176702223;-0.83202723134552303286;-0.80521326166877837949;0.82942452273928868323;0.8811644162344657305;0.78678643314580876122;-0.83938104318754469535;-0.88651798964236006029;0.84411750846543309734;0.82153084131522402522;-0.74727895226160012676;0.80608724461955583163;-0.79287129424292335234;-0.78600454258231722271;-0.90289785706433789425;0.82095930114567872149;-0.83538518392748339902;0.81177289344956649053;0.80905348585936764927;-0.76038124768746095938;0.75608083618887467203;-0.8114849626467098842;0.81831067838797133529;0.770997969150950202;0.77629421487410199543;-0.82107917481430725548;-0.77949833318115213654;0.77353618192905659878;-0.81746880343623806731;0.73978541617524762319;-0.7686462151108616192;-0.74073802558617385383;-0.75362665235956083354;0.78209150289280016111;0.79610130357608255292;0.71898215381117680511;0.75283988558562831006;-0.69233376512924138524;-0.73326021731338186793;0.76617390030997190031;0.66842189603423030064;-0.70495934267009641161;0.76039114657671058595;-0.68588801255730003081;0.71717750760040499536;0.71409254506358577075;0.70314058713216809338;0.69826741849017426311;0.70266679665589648618;0.72944442737477388494;0.74615378254363073918;0.75935876697847803918;0.68460969174236729895;-0.74661131709456041161;0.70397431731857218473;-0.70440119504375042769;-0.6990738675905074917;0.65691412137409099792;0.74168615522177150368;-0.69043512446506194458;-0.6697433163948615098;-0.67716316650075714101;0.70269693141756828236;-0.69729259852926228636;-0.6990134525148939959;0.67206110911295380284;-0.6616577731309475352;0.59983023048466432847;-0.69337067567378640742;0.64773891551850626414;-0.64103750831008710076;-0.64205723172847228852;0.73056812423356909392;-0.62318568792611706009;0.62762340064600419254;-0.64338660340453557307;-0.60294015747833351604;0.62182044108117662251;0.58174197488732282135;-0.65235455044689005799;-0.57435773412273993799;-0.60251671461062372703;-0.51578177455102802718;0.57067001502699032933;0.65373856856966161555;-0.62531485216438420505;0.51099994727639630465;0.64356617625089929113;-0.63111967258137313674;0.5601443928065498179;-0.58436401823361716623;0.63861592458259464067;-0.58438350393997451349;-0.59881036169825596627;0.60706686845469515212;0.50597720711451521503;-0.54699016220753127193;-0.51366274026294023169;0.58614159854927005977;-0.55790755610382314345;-0.59743031347282760102;0.56394949818704975808;0.56871654625611545164;-0.55122265199718678552;0.59660347184260420494;0.47331406496461120303;0.542503380400758628;-0.50336083564074018959;0.60896399787295729666;-0.49380131940552030034;-0.49850587695935205668;0.53042467952937022435;0.52432282425061393916;0.45301723943668487093;-0.49865791569395517824;0.45657215605866102726;-0.35973033709827123561;-0.5138126308202083381;0.56056805490227845645;-0.5506148398670762667;-0.45262316041783373644;0.43969119967037478824;-0.49842048106108449712;-0.49636751109170240692;0.47064500135262710145;-0.46562950156119764511;0.46608320079620896292;0.44605656569278367973;-0.53159018791686674099;0.48145219469961197944;-0.4869257308685846497;-0.4052499650747322657;-0.42941934513251173167;0.52332528178484361003;0.52998049672456337689;0.50150071145828756247;-0.41391411956413071138;0.45942892440604105042;0.41557121415704351142;-0.42186623015188212671;-0.40144379632364179944;-0.44807300004809852423;0.47565754395900966545;0.43119445199228073617;0.37179121944474674555;0.40435709326353691528;-0.4528740401790097847;-0.48064082320823209304;-0.35538416417251844637;0.40434128752019043995;0.42202589057277056783;-0.31597114308396451188;-0.40587032637118125589;0.40742178881412183244;-0.45120016692709946637;-0.38843247610548814475;-0.46332307166851532942;-0.33832260344342851566;0.3449677771465478604;0.29992362458364557609;0.33940384275877472087;0.3974593015561529441;0.34170265317377801484;0.34389203217372937837;-0.36909444003812530433;0.42469661909906769592;0.33242153182685468815;0.31400804542857718715;0.41319655918826386776;-0.34642390251331639517;-0.32078982755015778006;-0.33496869112719690476;0.30948152700205816545;0.28775948254883204624;-0.27743814024414920061;-0.27783208670611159175;0.37307566318452539678;-0.37915172907839733707;0.40626600576743771009;0.30132088058914352491;-0.27341956002723771801;0.29974341529696496567;0.25765247296637538987;0.36211772162783145701;0.34633563805969974636;0.26511060962370303073;-0.29042944753264565705;0.32072027726241875145;0.33172340159085339506;-0.30339114445064546688;0.33503804549220272957;0.31642376818950757045;-0.14542826001614558851;0.27630583978626493957;-0.1923714972602338924;0.27577035678392203266;-0.27597224847838625061;0.35492034944184114309;0.14836800741874703324;-0.19914786997602110241;0.25815923650792843569;-0.27076665946945183139;-0.2408397850927463113;0.22037063956862448788;0.14038728289760493695;0.24327453136921103893;-0.21972808781092878494;0.27976782465603683425;-0.23899993448084924652;-0.25858936676533517129;0.24202211305648377304;-0.1911389443554919465;-0.18076423958949383586;0.19315540735107974979;0.29969111745012816383;0.16357260773456150371;-0.18891150395860126232;0.25795480303290330149;-0.19348499648466441214;-0.17261731544185893172;-0.23449074777052544838;-0.19009506585851665417;0.21924845135661036322;-0.20875826111464892976;0.14320976289338580023;-0.27878481081488659665;-0.11253113358578926972;0.24403307941027882477;0.24844895609705494599;0.0025917772221413937911;0.1467284040197783801;0.10750641468040716964;-0.092726600282094090444;0.093040163946804019246;-0.14896310869284792866;-0.16440740779981086184;-0.14401615489983771168;-0.15862007869711339247;-0.17484534146313884806;-0.15427456890219856667;-0.068565620299020552109;-0.12234764202896944441;-0.2173848554302017233;-0.11189662824610248359;0.13526825942307929562;-0.16701618503155471473;0.027944598982046695013;-0.14682008740710100758;-0.084714753427196864966;0.13944224297812218838;0.20253450417369150971;0.15672901569440200742;-0.087109646686771013968;0.098916573685817174577;-0.12626442990803068978;0.073313444376364042543;-0.09638843455149853634;-0.1529590553477189474;-0.056081817089656725572;0.16805641453248032491;0.10370130853226680601;-0.10087901728485725028;-0.013302417511599100297;-0.096943096339712722664;0.062648326854401303887;0.070121720009060181655;-0.076418263458951876665;-0.0081589345230638470746;0.090787508841881400801;-0.004011104032074827748;0.052393514376530544996;-0.077082034847816033318;0.044799456997457527685;0.083463054558424440432;0.054167681301009025507;0.021984557802130620663;0.0063190947490359607869;0.043955539744645069633;-0.10943013840479792753;-0.029082944825191005267;0.0040731868794438479478;0.0077810956559671426089;0.027987758709622688469;0.020027006593918043192;0.016178268254552385236;-0.06007130563792450334;0.036544083143552952286;0.037450911061170946703;-0.00079178185668206217673;-0.028101733752766557689;0.019780342260152957573;-0.021822103886194751943;-0.040878912676171948182;-0.00046701665257987214118;-0.034484549119588064903;-0.031562104321784162364;0.00062832225889778272124;-0.042653193297163403963;0.0052602945377025981527;0.00092256489347192877862;-0.045407238202672658978;0.021423664449462188203;-0.036488749629630959359;-0.036827340717696117889;0.10225814598725589677;-0.050855745905837079768;0.09898206766205322471;0.15098296009079284485;-0.12327673009518352876;-0.13698451298641406426;0.15266717484609806932;0.035967195866243438507;-0.043121907768212279499;0.064565985219378480009;-0.09926484757532758052;-0.11956950878040097097;0.088337854640701082842;-0.081373102838369446022;-0.064549916485341904404;0.11311384622983342019;-0.020953463728481960815;-0.019905191450852378898;0.15904367582900030831;0.13334601293948741829;-0.00039582363105565999267;0.070119488569539131784;0.12277632218147072063;0.043562916666074084915;-0.13021751847692350501;-0.1077482163338707305;0.22758538752539542172;0.15226871125185803546;-0.15092956077166294615;0.20615860697820737046;0.13178650719809290037;0.10020216817095051187;0.14143782462442697212;-0.16815934444292054972;0.19630614843814156445;0.2016374889872493692;0.11849959339127792701;0.10592776713269475009;-0.27080389927770137648;0.19555748672965114765;-0.26735183628496539399;0.20378601892584730093;-0.25795863505579902375;0.17594294277502342538;-0.16415565775046048191;0.082812903309476479374;-0.15057258015834984466;-0.18325835400357134986;0.17031588694011440266;0.2272461556590470444;-0.24604363901122916491;0.19915107507270313203;0.29989481307706777891;0.23175078354845243811;0.22803887032852443451;-0.18824232094519158243;-0.20084548989719108736;0.25818146088826204876;0.29177103291630390736;0.22931899724869606039;-0.22689897086430918516;-0.29015219794135949094;-0.2773607037211135995;0.32309816100712701603;0.19692518687842378022;0.18070393477834612028;0.32114133205768319312;0.20770406798862656172;0.22216242953490075718;-0.24150754736377208487;0.2004599061259652315;-0.21140868903121207145;0.22275111079058687591;0.26264079198664019854;0.28889952031764970508;-0.25839654519013105505;-0.27281431288046426031;0.25696234535843570645;0.27418145765621138032;-0.34345495769225159455;-0.28445646283852554781;-0.28331206359882232793;-0.33436402967788342311;0.27601053665260560921;0.23258429489885681929;0.38169453738844266732;-0.2770918915071074462;0.23556247227522511545;-0.27980273007076378322;0.36939917365658242465;0.24343189471683246428;0.35918158804458483191;-0.34950047992193944424;-0.36832559715304219461;0.29093367518771012792;-0.40217641042602825241;-0.35495889196195662407;-0.27793310974396456814;0.32644079677091403235;-0.36205668829868387082;0.40606822898389005605;-0.41520341148141243215;0.35148223897126706916;0.35227563524056682009;-0.26472818201593434351;0.36250333769664688788;0.32909747462011701424;0.33496118245564926452;0.39500124412742543267;0.48219019571752641529;0.33144288063390309196;0.30802255131789957332;0.3789820499009320387;-0.30231673730169250636;-0.45274308508499910841;0.47164984719899205601;-0.45254888770042539514;-0.4133681922862388336;0.45254362053919783282;-0.39285217748104112045;-0.42552022690570623364;0.39075628874975443239;-0.38275296348739551444;0.37846657694886737255;0.46334469193266542009;0.37644807450370881829;0.53782055917734861161;0.46444078627804297188;0.47554256976396330758;0.42925495204971780838;-0.46370528534944682741;-0.5099126350811051811;0.41741323785661932355;-0.42782802528773472561;0.48665489390798954972;-0.43946758367165644765;0.45221584577746143374;-0.49196099565170653412;0.503849279411527462;-0.43199890386429568023;0.48244268584033178637;0.45340208688925159874;0.49771319457623808269;0.48059198419724413753;0.54394714868151639031;-0.5511441768357834281;0.50503012668842894861;0.46268178555430028442;-0.47953698078234707758;-0.50533152930517166723;0.4314957159894628469;0.50061003481260335501;-0.46088145620566350891;-0.48102325904553877578;-0.62005590042909253246;0.52713846885793247399;-0.42660223112650652588;0.52905640106940199363;0.54772690132122303641;-0.62237154380730752568;-0.54882164606117178618;0.5537772185500756672;-0.50406182469574067895;-0.57596209080189642027;0.50358670807158312233;-0.56355365902571241676;0.4653369358787381449;0.60707971362739221455;0.58383274665662221459;0.51876374588663487497;0.53078472785923880295;-0.55424139317161680296;0.5279457749795650745;0.59295563575735865669;0.54955278810798791067;-0.61876757239020208079;-0.57118908676879254216;0.55118241833283854181;-0.58025733951781344189;-0.60564568968450549491;-0.60475652918541222824;0.6095273325956253041;0.57749842039515364434;0.5867555737616262368;-0.60119502520586354244;0.65455736049464674498;0.6150183052162357944;0.60151272370586117422;-0.58789334560543560215;0.64323270363955886708;-0.68165615548925229117;0.62785560265133832925;0.62381281637251850825;-0.62209312898526702984;-0.61173128346504046515;0.57341385188865157119;-0.64439523560168265881;-0.66793512261858944967;-0.64095282542473863696;-0.63694216562529260006;-0.61935349632615877002;-0.66103108457558912292;0.70413681737541378158;-0.70479352065694733653;-0.65959286713298947991;0.64641738459888842261;0.64749160306078112903;0.65493472060459934081;0.68093634134847358386;-0.68630525942515008264;-0.70761219979340461705;-0.64923022123328166799;-0.73160540389398653893;0.70051672633156725212;-0.66863725138246576929;-0.6944177509985418606;-0.73921510180128269241;0.6336321647304434812;0.66875238897884725642;0.72808142086785609948;-0.723091264774181397;0.73703561593069000235;0.72598342874077514963;0.67893853682363092261;0.74215266927390954876;0.74111954409685476541;0.75787371840827233438;-0.68508211191500911763;0.75856160061120891935;0.70944461461967966986;-0.6894206548677908053;-0.71499934373622864303;0.73575553166959029028;-0.81089364651494555414;0.76813540377387623526;0.7467960896462966236;0.72962665959372652313;0.73897741729999166793;0.78117058653367199561;0.81798743974660981237;0.72920576239100620697;0.80116494589653741709;0.75308149729037976439;-0.74319953918826731076;0.7350181666330573016;-0.81202717141864699801;-0.81422426846663087385;-0.79942361931530192543;0.84463163142464936506;-0.76340423192802175922;0.81914667675830199478;0.78719903456689244159;0.78730948595364436127;-0.74333673301904823472;0.76324729969854776535;0.83234818523443754312;0.81806505729879563926;0.79672901834958786971;0.84022667117157534555;-0.86015693210906785104;-0.83648084313282433655;-0.81850790132219464201;0.79965754575773240198;0.86139127606584142693;-0.86237711896928070932;-0.82652292417478256947;0.81848450720211773923;-0.80265888168307297779;0.85233849531453165227;-0.83944810014675297349;0.86332138571756977807;-0.81351522772119466698;-0.83399740232195074263;-0.8529802255921106191;-0.89057462742830795221;0.88358801043982659884;-0.85331592602825911964;0.94032658503668742345;-0.87208363725682513934;-0.9038177023743980687;-0.8452161785333561772;-0.8358427598225146582;-0.89436305287364736838;-0.84587193166339846861;0.92140363724601737072;-0.90073058707337028128;-0.88159528553896415737;0.91740951487212152315;-0.9115035904020023505;-0.90950380842810141235;0.90091045590782437014;-0.93143495200068893158;-0.93671020637581503543;0.91745978101005642991;0.91601487666380987385;0.93883161902614387095;0.92522564830271059044;-0.96066638886280064646;-0.99700156717442078858;0.93419423767887366239;0.92913812377059612491;0.91092455392544635639;-1.0038774593553383419;-0.94461658736959408511;-0.98456812600627463539;0.98691287818701756152;0.97048053254017596636;0.97848199100987687249;-0.96702069032006521265;1.0075789011493636593;0.95481945225821540646;0.96988780711484623431;0.99267815486301935746;-0.97265759909589233345;0.97340340859905427617;-1.0362405728591110865;-0.96663115191663084858;-1.0006015424917986678;-1.0000488733776979888;-1.0144934808408836435;-0.99830590088173543073;-0.99821158805005882897;-1.0195325227218769992;-1.0229942171606059009;-1.0213052584705561987;-1.0153448625510566039;0.99016341313829225879;-1.0332872328773892168;1.0276586388683330853;-1.0109533913308146325;-1.0326024415702661496;-1.0204571408174236691;-1.0505270863680007398;1.0615877441679790749;1.0193625328540814046;1.0538046016139079875;1.0366144119386766498;1.0333855389083623777;1.0266518840187015549;1.0167474798463864705;-1.0412672029431584431;1.0727135477879261583;1.0424031362952645985;-1.0510701453556308138;1.0833670454477268841;1.0567936398698880573;-1.0819277599770817933;1.1099191455718575572;-1.0665857207831870301;-1.0809206307971030991;1.1357224427426917401;-1.0633369872573692483;1.1067539038446378985;-1.1091314480514160579;-1.0873469167588336592;1.0839338094156041326;-1.1040436824666250981;1.1068955094190779942;-1.0983999671712245672;-1.1114271613545607664;1.0885124624840121221;-1.114936468091233035;1.0657652474841614598;-1.1146808695910686016;-1.1385456959251369913;1.1174331557668197856;-1.1372288011765370186;1.1441990165245716771;1.1675700617364348499;1.204472571433529815;-1.1440763323392844075;-1.1443415620160184432;-1.1586561709101796946;-1.1483014274388840814;1.1772181040238456351;1.1395533530862700378;1.2228759481240607521;-1.1515043260094468547;1.1728723523843540555;-1.1855629163518544011;1.1591271608642281787;-1.1716264915042609029;-1.1577708292005945978;-1.1800130243609054936;-1.1737618723176892477;1.2289431071689802533;1.2221084051910093837;1.203690609400883238;-1.1836989444080778799;-1.2137543881173744609;-1.2295404021512288395;-1.1846022824796060835;1.2369050419912610561;-1.2003376357750221715;-1.2340705586263571725;-1.2231864022894196609;1.2377713529466625175;1.2088522673050723366;-1.2108760968190912521;1.2398938229710687597;-1.2264268012711698752;-1.2115340943321126765;-1.2333851921425731391;1.2460357584202539361;1.2625047997685014955;1.2355142357587893809;-1.2374018698849345022;-1.2329689134709669318;1.2597191573635420081;1.2735126856730705214;1.3181096501483118288;-1.2809584686495292249;-1.2646776018009229237;-1.2709143300070246951;1.2625835535803706655;-1.3133429824679596631;1.3112800847879348964;1.2207609608233678244;-1.2974106328142969158;1.2684097013603179427;1.2943861201578827913;-1.2862200977766291565;1.3109686522003090037;-1.3253231758952968011;1.2947728489580783151;-1.3351554752946035443;-1.3232405958039079596;-1.310613708582621495;-1.3017243752790790268;-1.2949514735840426116;1.3498935721426301715;1.3538198883273171447;1.3231837466750597798;-1.3381102914131626225;-1.2995834413177427269;-1.3460949748135380144;1.33913223095694911;1.3411378498659383141;-1.3786240010772219389;-1.3385714505584243028;-1.3344457566498977297;1.3358906313515033215;-1.3551545614985942567;-1.3269044183324711916;-1.4014321551529156373;-1.3712892438516828264;1.3882705020071770541;1.3962731595606920898;1.3529651577415482411;-1.3599347970781610062;-1.3923108063277684909;-1.3580659975087852054;-1.3818050704702873333;-1.3702658912720497852;1.3739603623790479503;1.3901867600023436466;1.4177889258228253411;1.3560452585566984407;1.3724986188688861333;1.3859193157133162799;1.4258454301630953953;-1.4220225423792645092;-1.4164646312315447574;1.4315888538365268978;1.4073058764526702458;1.4466498339725701872;-1.4285434759431880902;-1.4455065428650915038;-1.411343840043517428;-1.431967277931228022;-1.4542457871288643201;1.4504514764803673188;-1.4769985430313301578;1.4573656983159477551;-1.4657071436054325275;1.4668911141498617301;1.4629485999812872343;-1.4724324790119167616;-1.4948168789718292437;-1.470833829192151887;1.4405690096516325927;1.4614107527231743333;1.4435955381648641893];

% Layer 2
b2 = [-0.50385258295307611132;0.39421984967453777005;-0.47689472568496549254;0.090139689217200766325;0.86637810988585395311;0.92826066336657364619;-0.47732001439385551844;0.053570221830526690288;-0.050801587392779408436;-0.30420185555097550845];

% ===== SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if ~isCellX
    X = {X};
end

% Dimensions
TS = size(X,2); % timesteps
if ~isempty(X)
    Q = size(X{1},1); % samples/series
else
    Q = 0;
end

% Allocate Outputs
Y = cell(1,TS);

% Time loop
for ts=1:TS
    
    % Input 1
    X{1,ts} = X{1,ts}';
    Xp1 = mapminmax_apply(X{1,ts},x1_step1);
    
    % Layer 1
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*Xp1);
    
    % Layer 2
    a2 = softmax_apply(repmat(b2,1,Q) + LW2_1*a1);
    
    % Output 1
    Y{1,ts} = a2;
    Y{1,ts} = Y{1,ts}';
end

% Final Delay States
Xf = cell(1,0);
Af = cell(2,0);

% Format Output Arguments
if ~isCellX
    Y = cell2mat(Y);
end
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