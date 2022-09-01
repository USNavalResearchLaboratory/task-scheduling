Search.setIndex({docnames:["index","task_scheduling","task_scheduling.algorithms","task_scheduling.generators","task_scheduling.mdp"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","task_scheduling.rst","task_scheduling.algorithms.rst","task_scheduling.generators.rst","task_scheduling.mdp.rst"],objects:{"":[[1,0,0,"-","task_scheduling"]],"task_scheduling.algorithms":[[2,0,0,"-","base"],[2,0,0,"-","wrappers"]],"task_scheduling.algorithms.base":[[2,1,1,"","branch_bound"],[2,1,1,"","branch_bound_priority"],[2,1,1,"","brute_force"],[2,1,1,"","earliest_drop"],[2,1,1,"","earliest_release"],[2,1,1,"","mcts"],[2,1,1,"","priority_sorter"],[2,1,1,"","random_sequencer"]],"task_scheduling.algorithms.wrappers":[[2,1,1,"","ensemble_scheduler"],[2,1,1,"","sort_wrapper"]],"task_scheduling.base":[[1,2,1,"","RandomGeneratorMixin"],[1,2,1,"","SchedulingProblem"],[1,2,1,"","SchedulingSolution"],[1,1,1,"","get_now"]],"task_scheduling.base.RandomGeneratorMixin":[[1,3,1,"","make_rng"],[1,4,1,"","rng"]],"task_scheduling.base.SchedulingProblem":[[1,5,1,"","ch_avail"],[1,5,1,"","tasks"]],"task_scheduling.base.SchedulingSolution":[[1,5,1,"","loss"],[1,5,1,"","sch"],[1,5,1,"","t_run"]],"task_scheduling.generators":[[3,0,0,"-","channels"],[3,0,0,"-","problems"],[3,0,0,"-","tasks"]],"task_scheduling.generators.channels":[[3,2,1,"","Base"],[3,2,1,"","BaseIID"],[3,2,1,"","Deterministic"],[3,2,1,"","UniformIID"]],"task_scheduling.generators.channels.Base":[[3,3,1,"","__call__"],[3,3,1,"","summary"]],"task_scheduling.generators.channels.BaseIID":[[3,3,1,"","__call__"]],"task_scheduling.generators.channels.Deterministic":[[3,3,1,"","__call__"],[3,3,1,"","from_uniform"],[3,3,1,"","summary"]],"task_scheduling.generators.channels.UniformIID":[[3,3,1,"","summary"]],"task_scheduling.generators.problems":[[3,2,1,"","Base"],[3,2,1,"","Dataset"],[3,2,1,"","DeterministicTasks"],[3,2,1,"","FixedTasks"],[3,2,1,"","PermutedTasks"],[3,2,1,"","Random"]],"task_scheduling.generators.problems.Base":[[3,3,1,"","__call__"],[3,3,1,"","summary"]],"task_scheduling.generators.problems.Dataset":[[3,3,1,"","add"],[3,3,1,"","load"],[3,4,1,"","n_problems"],[3,3,1,"","shuffle"],[3,3,1,"","split"],[3,5,1,"","stack"],[3,3,1,"","summary"]],"task_scheduling.generators.problems.DeterministicTasks":[[3,5,1,"","cls_task_gen"]],"task_scheduling.generators.problems.FixedTasks":[[3,5,1,"","cls_task_gen"],[3,3,1,"","continuous_linear_drop"],[3,3,1,"","discrete_linear_drop"],[3,4,1,"","solution"]],"task_scheduling.generators.problems.PermutedTasks":[[3,5,1,"","cls_task_gen"]],"task_scheduling.generators.problems.Random":[[3,3,1,"","continuous_exp"],[3,3,1,"","continuous_linear"],[3,3,1,"","continuous_linear_drop"],[3,3,1,"","discrete_linear"],[3,3,1,"","discrete_linear_drop"]],"task_scheduling.generators.tasks":[[3,2,1,"","Base"],[3,2,1,"","BaseIID"],[3,2,1,"","ContinuousUniformIID"],[3,2,1,"","Dataset"],[3,2,1,"","Deterministic"],[3,2,1,"","DiscreteIID"],[3,2,1,"","Fixed"],[3,2,1,"","GenericIID"],[3,2,1,"","Permutation"]],"task_scheduling.generators.tasks.Base":[[3,3,1,"","__call__"],[3,3,1,"","summary"]],"task_scheduling.generators.tasks.BaseIID":[[3,3,1,"","__call__"]],"task_scheduling.generators.tasks.ContinuousUniformIID":[[3,3,1,"","exp"],[3,3,1,"","linear"],[3,3,1,"","linear_drop"],[3,3,1,"","summary"]],"task_scheduling.generators.tasks.Dataset":[[3,3,1,"","__call__"],[3,3,1,"","add_tasks"],[3,3,1,"","shuffle"]],"task_scheduling.generators.tasks.Deterministic":[[3,3,1,"","__call__"]],"task_scheduling.generators.tasks.DiscreteIID":[[3,3,1,"","linear_drop_uniform"],[3,3,1,"","linear_uniform"],[3,3,1,"","summary"]],"task_scheduling.generators.tasks.Fixed":[[3,3,1,"","__call__"],[3,3,1,"","continuous_linear_drop"],[3,3,1,"","discrete_linear_drop"]],"task_scheduling.generators.tasks.GenericIID":[[3,3,1,"","linear_drop"]],"task_scheduling.generators.tasks.Permutation":[[3,3,1,"","__call__"]],"task_scheduling.mdp":[[4,0,0,"-","base"],[4,0,0,"-","environments"],[4,0,0,"-","features"],[4,0,0,"-","reinforcement"],[4,0,0,"-","supervised"]],"task_scheduling.mdp.base":[[4,2,1,"","Base"],[4,2,1,"","BaseLearning"],[4,2,1,"","RandomAgent"]],"task_scheduling.mdp.base.Base":[[4,3,1,"","__call__"],[4,3,1,"","predict"],[4,3,1,"","summary"]],"task_scheduling.mdp.base.BaseLearning":[[4,3,1,"","learn"],[4,4,1,"","learn_params"],[4,3,1,"","reset"],[4,3,1,"","summary"]],"task_scheduling.mdp.base.RandomAgent":[[4,3,1,"","predict"]],"task_scheduling.mdp.environments":[[4,2,1,"","Base"],[4,2,1,"","Index"],[4,1,1,"","int_to_seq"],[4,1,1,"","seq_to_int"]],"task_scheduling.mdp.environments.Base":[[4,4,1,"","ch_avail"],[4,3,1,"","close"],[4,3,1,"","get_problem_spaces"],[4,3,1,"","infer_valid_mask"],[4,4,1,"","n_ch"],[4,4,1,"","n_features"],[4,4,1,"","n_tasks"],[4,3,1,"","obs"],[4,3,1,"","opt_action"],[4,3,1,"","opt_rollouts"],[4,4,1,"","problem_gen"],[4,3,1,"","render"],[4,3,1,"","reset"],[4,3,1,"","seed"],[4,4,1,"","sorted_index"],[4,4,1,"","sorted_index_inv"],[4,3,1,"","step"],[4,3,1,"","summary"],[4,4,1,"","tasks"]],"task_scheduling.mdp.environments.Index":[[4,3,1,"","infer_valid_mask"],[4,3,1,"","opt_action"]],"task_scheduling.mdp.features":[[4,1,1,"","encode_discrete_features"],[4,1,1,"","normalize"],[4,1,1,"","param_features"]],"task_scheduling.mdp.reinforcement":[[4,2,1,"","MultiExtractor"],[4,2,1,"","StableBaselinesScheduler"],[4,2,1,"","ValidActorCriticPolicy"],[4,2,1,"","ValidDQNPolicy"],[4,2,1,"","ValidQNetwork"]],"task_scheduling.mdp.reinforcement.MultiExtractor":[[4,3,1,"","cnn"],[4,3,1,"","forward"],[4,3,1,"","mlp"],[4,5,1,"","training"]],"task_scheduling.mdp.reinforcement.StableBaselinesScheduler":[[4,4,1,"","env"],[4,3,1,"","learn"],[4,3,1,"","load"],[4,3,1,"","make_model"],[4,5,1,"","model_defaults"],[4,3,1,"","predict"],[4,3,1,"","reset"],[4,3,1,"","save"]],"task_scheduling.mdp.reinforcement.ValidActorCriticPolicy":[[4,3,1,"","evaluate_actions"],[4,3,1,"","forward"],[4,3,1,"","get_distribution"],[4,5,1,"","training"]],"task_scheduling.mdp.reinforcement.ValidDQNPolicy":[[4,3,1,"","make_q_net"],[4,5,1,"","training"]],"task_scheduling.mdp.reinforcement.ValidQNetwork":[[4,3,1,"","forward"],[4,5,1,"","training"]],"task_scheduling.mdp.supervised":[[4,2,1,"","BasePyTorch"],[4,2,1,"","BaseSupervised"],[4,2,1,"","LitModel"],[4,2,1,"","LitScheduler"],[4,2,1,"","TorchScheduler"]],"task_scheduling.mdp.supervised.BasePyTorch":[[4,3,1,"","from_gen"],[4,3,1,"","load"],[4,3,1,"","predict"],[4,3,1,"","predict_prob"],[4,3,1,"","reset"],[4,3,1,"","save"]],"task_scheduling.mdp.supervised.BaseSupervised":[[4,3,1,"","learn"],[4,3,1,"","predict"],[4,3,1,"","reset"],[4,3,1,"","train"]],"task_scheduling.mdp.supervised.LitModel":[[4,3,1,"","configure_optimizers"],[4,3,1,"","forward"],[4,5,1,"","training"],[4,3,1,"","training_step"],[4,3,1,"","validation_step"]],"task_scheduling.mdp.supervised.LitScheduler":[[4,3,1,"","from_gen_mlp"],[4,3,1,"","from_gen_module"],[4,3,1,"","from_module"],[4,3,1,"","mlp"],[4,3,1,"","reset"],[4,3,1,"","train"]],"task_scheduling.mdp.supervised.TorchScheduler":[[4,5,1,"","device"],[4,3,1,"","from_gen_mlp"],[4,3,1,"","mlp"],[4,3,1,"","train"]],"task_scheduling.nodes":[[1,2,1,"","MCTSNode"],[1,2,1,"","ScheduleNode"],[1,2,1,"","ScheduleNodeBound"],[1,2,1,"","ScheduleNodeReform"]],"task_scheduling.nodes.MCTSNode":[[1,3,1,"","backup"],[1,4,1,"","children"],[1,3,1,"","evaluation"],[1,3,1,"","expansion"],[1,4,1,"","is_leaf"],[1,4,1,"","is_root"],[1,4,1,"","l_avg"],[1,4,1,"","n_tasks"],[1,4,1,"","n_visits"],[1,4,1,"","parent"],[1,3,1,"","select_child"],[1,3,1,"","selection"],[1,4,1,"","seq"],[1,4,1,"","seq_rem"],[1,3,1,"","update_stats"],[1,4,1,"","weight"]],"task_scheduling.nodes.ScheduleNode":[[1,3,1,"","branch"],[1,3,1,"","brute_force"],[1,4,1,"","ch_avail"],[1,3,1,"","earliest_drop"],[1,3,1,"","earliest_release"],[1,4,1,"","loss"],[1,3,1,"","mcts"],[1,4,1,"","n_ch"],[1,4,1,"","n_tasks"],[1,3,1,"","priority_sorter"],[1,3,1,"","roll_out"],[1,4,1,"","sch"],[1,4,1,"","seq"],[1,3,1,"","seq_append"],[1,3,1,"","seq_extend"],[1,4,1,"","seq_rem"],[1,3,1,"","summary"],[1,4,1,"","tasks"]],"task_scheduling.nodes.ScheduleNodeBound":[[1,4,1,"","bounds"],[1,3,1,"","branch_bound"],[1,3,1,"","branch_bound_priority"],[1,4,1,"","l_lo"],[1,4,1,"","l_up"],[1,3,1,"","seq_append"],[1,3,1,"","seq_extend"]],"task_scheduling.nodes.ScheduleNodeReform":[[1,3,1,"","reform"]],"task_scheduling.results":[[1,1,1,"","evaluate_algorithms_gen"],[1,1,1,"","evaluate_algorithms_single"],[1,1,1,"","evaluate_algorithms_train"]],"task_scheduling.spaces":[[1,2,1,"","DiscreteMasked"],[1,2,1,"","DiscreteSet"],[1,2,1,"","Permutation"],[1,1,1,"","broadcast_to"],[1,1,1,"","concatenate"],[1,1,1,"","get_space_lims"],[1,1,1,"","reshape"],[1,1,1,"","stack"]],"task_scheduling.spaces.DiscreteMasked":[[1,3,1,"","contains"],[1,4,1,"","mask"],[1,4,1,"","n"],[1,3,1,"","sample"],[1,4,1,"","valid_entries"]],"task_scheduling.spaces.DiscreteSet":[[1,3,1,"","add_elements"],[1,3,1,"","contains"],[1,3,1,"","sample"]],"task_scheduling.spaces.Permutation":[[1,3,1,"","contains"],[1,3,1,"","sample"]],"task_scheduling.tasks":[[1,2,1,"","Base"],[1,2,1,"","Exponential"],[1,2,1,"","Generic"],[1,2,1,"","Linear"],[1,2,1,"","LinearDrop"],[1,2,1,"","PiecewiseLinear"],[1,2,1,"","ReformMixin"]],"task_scheduling.tasks.Base":[[1,3,1,"","__call__"],[1,5,1,"","param_names"],[1,4,1,"","params"],[1,4,1,"","plot_lim"],[1,3,1,"","plot_loss"],[1,3,1,"","summary"],[1,3,1,"","to_series"]],"task_scheduling.tasks.Exponential":[[1,3,1,"","__call__"],[1,5,1,"","param_names"],[1,3,1,"","reform_param_lims"]],"task_scheduling.tasks.Generic":[[1,3,1,"","__call__"],[1,5,1,"","param_names"]],"task_scheduling.tasks.Linear":[[1,5,1,"","param_names"],[1,5,1,"","prune"],[1,4,1,"","slope"]],"task_scheduling.tasks.LinearDrop":[[1,4,1,"","l_drop"],[1,5,1,"","param_names"],[1,5,1,"","prune"],[1,3,1,"","reform_param_lims"],[1,4,1,"","slope"],[1,4,1,"","t_drop"]],"task_scheduling.tasks.PiecewiseLinear":[[1,3,1,"","__call__"],[1,4,1,"","corners"],[1,5,1,"","param_names"],[1,4,1,"","plot_lim"],[1,5,1,"","prune"]],"task_scheduling.tasks.ReformMixin":[[1,3,1,"","reform_param_lims"],[1,3,1,"","reparam"],[1,3,1,"","shift"]],"task_scheduling.util":[[1,1,1,"","check_schedule"],[1,1,1,"","eval_wrapper"],[1,1,1,"","evaluate_schedule"],[1,1,1,"","plot_losses_and_schedule"],[1,1,1,"","plot_schedule"],[1,1,1,"","plot_task_losses"],[1,1,1,"","summarize_tasks"]],task_scheduling:[[2,0,0,"-","algorithms"],[1,0,0,"-","base"],[3,0,0,"-","generators"],[4,0,0,"-","mdp"],[1,0,0,"-","nodes"],[1,0,0,"-","results"],[1,0,0,"-","spaces"],[1,0,0,"-","tasks"],[1,0,0,"-","util"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","property","Python property"],"5":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:property","5":"py:attribute"},terms:{"0":[1,2,3,4],"00028":4,"01":4,"02":4,"1":[1,3,4],"10":4,"12":[1,3],"16":4,"1704":4,"1e":[1,4],"2":[1,3,4],"3":[1,3,4],"35":3,"4":3,"5":[3,4],"50":3,"6":[3,4],"718281828459045":1,"99":4,"abstract":[1,3,4],"boolean":1,"case":4,"class":[1,3,4],"default":[1,2,4],"do":4,"float":[1,2,3,4],"function":[1,2,4],"import":1,"int":[1,2,3,4],"new":[1,3,4],"return":[1,2,3,4],"static":[1,4],"super":4,"switch":4,"true":[1,2,4],"while":4,A:[1,4],And:4,At:4,But:4,By:4,If:[1,2,4],In:4,It:[0,4],The:[1,4],There:4,__call__:[1,3,4],_param_gen:3,a2c:4,a_lim:3,ab:4,abc:[1,3,4],abov:[1,4],ac:4,acc:4,access:3,accident:4,accord:4,accuraci:4,achiev:4,act:4,action:[1,4],activ:0,actor:4,actorcriticpolici:4,actual:4,adam:4,add:[1,3,4],add_el:1,add_imag:4,add_task:3,addit:[1,4],advanc:1,after:4,afterward:4,against:1,agent:4,algorithm:[0,1,4],alia:[1,3],all:[1,2,3,4],allot:[1,2],allow:[1,2,3],along:[1,4],also:4,although:4,an:[1,4],ancestor:1,ani:[1,4],ansi:4,anyth:4,append:[1,4],approach:1,ar:[2,3,4],aren:4,arg:4,argmax:4,argument:[1,3,4],arrai:[1,4],array_lik:4,arxiv:4,assess:0,associ:4,assum:[1,3],attribut:1,automat:[1,4],auxiliari:4,avail:[1,2,3,4],averag:[1,4],ax:1,ax_kwarg:1,axi:1,b:1,b_lim:3,back:4,backprop:4,backup:1,backward:4,bar:4,base:[0,3],baseenv:4,basefeaturesextractor:4,baseiid:3,baselearn:4,baselines3:4,basepytorch:4,basesupervis:4,basic:4,batch:4,batch_idx:4,been:[1,4],being:4,below:4,best:2,between:4,bigint:4,binari:4,bit:4,bonu:1,bool:[1,2,3,4],both:0,bound:[1,2,3,4],bounded:1,branch:[1,2,3,4],branch_bound:[1,2],branch_bound_prior:[1,2],broadcast:1,broadcast_to:1,brute_forc:[1,2],c_explor:[1,2],calcul:[1,4],call:[3,4],callabl:[1,2,3,4],callback:4,can:[1,4],captur:4,care:4,carlo:[1,2],ch_avail:[1,2,3,4],ch_avail_gen:3,ch_avail_lim:[1,3],chain:2,channel:[0,1,2,4],check:[1,4],check_input:4,check_schedul:1,check_valid:1,child:1,children:[1,2],choic:1,choos:4,cl:4,classic:2,classmethod:[3,4],cleanup:4,close:4,closur:4,cls_task:3,cls_task_gen:3,cnn:4,cnn_kwarg:4,code:1,collect:[1,2,3,4],color:4,combin:1,common:[1,4],compar:1,complet:[1,2,3,4],comput:4,concaten:1,condit:[1,4],configur:4,configure_optim:4,constant:1,construct:[3,4],consumpt:4,contain:[1,4],continu:[3,4],continuous_exp:3,continuous_linear:3,continuous_linear_drop:3,continuousuniformiid:3,control:4,convent:4,core:[1,4],corner:1,correl:4,correspond:4,cosineann:4,could:4,count:1,cpu:4,creat:[1,2,3,4],critic:4,crm:0,cross_entropi:4,current:[1,4],custom:4,cycl:4,d:[3,4],data:[1,3,4],datafram:1,dataload:4,dataloader_idx:4,dataset:[3,4],debug:4,decid:4,decod:4,decreas:2,deepspe:4,def:4,defin:[1,4],depend:1,dequ:3,descend:1,describ:[1,4],detail:1,determin:2,determinist:[3,4],deterministictask:3,develop:0,devic:4,diagnost:4,dict:[1,3,4],dictionari:[3,4],differ:[1,4],dim:4,dis_opt:4,dis_sch:4,disabl:4,discret:[1,3,4],discrete_linear:3,discrete_linear_drop:3,discreteiid:3,discretemask:1,discreteset:[1,4],displai:4,distribut:[3,4],don:4,dot:1,dqn:4,dqnpolici:4,drop:[1,2],due:1,durat:1,duration_lim:3,duration_v:3,dure:3,e:4,each:[1,4],earliest:[1,2],earliest_drop:[1,2],earliest_releas:[1,2],element:[1,3,4],elif:4,els:4,enabl:[1,2,3,4],encod:4,encode_discrete_featur:4,end:4,enforc:4,ensemble_schedul:2,ensur:4,entropi:4,env:4,env_cl:4,env_param:4,environ:[0,1],episod:4,epoch:4,equal:4,escap:4,estim:4,eval:4,eval_wrapp:1,evalu:[1,2,4],evaluate_act:4,evaluate_algorithms_gen:1,evaluate_algorithms_singl:1,evaluate_algorithms_train:1,evaluate_schedul:1,everi:[1,4],exampl:4,example_imag:4,except:4,execut:[1,2,4],exhaust:[1,2],exist:1,exit:4,exp:3,expans:[1,2],experi:4,explicit:1,explor:[1,2],expon:1,exponenti:[1,3],exponentiallr:4,extend:1,extractor:4,factori:4,fals:[1,2,3,4],fancier:4,featur:[0,1],field:[1,4],fig_kwarg:1,figur:1,file:[1,3,4],file_path:3,first:[3,4],fit:4,fix:3,fixedtask:3,former:4,formul:4,forward:4,found:4,frame:4,framework:0,frequenc:4,frequent:[1,2],from:[1,3,4],from_gen:4,from_gen_mlp:4,from_gen_modul:4,from_modul:4,from_uniform:3,func:[1,2,4],futur:3,g:4,gan:4,garbag:4,gen_opt:4,gen_sch:4,gener:[0,1,2,4],genericiid:3,get:[1,4],get_distribut:4,get_now:1,get_problem_spac:4,get_space_lim:1,given:[2,4],goe:4,gpu:4,gradient:4,grid:4,guarante:1,gym:[1,3,4],ha:[1,4],handl:4,have:4,help:4,here:4,heterogen:3,heurist:[1,2],hidden:4,hidden_sizes_ch:4,hidden_sizes_joint:4,hidden_sizes_task:4,higher:[1,2],hook:4,how:4,http:[0,4],human:4,i:3,ident:3,ignor:4,imag:4,img_path:1,implement:[0,4],improv:4,includ:[0,1,4],incur:1,independ:3,index:[0,1,2,4],indic:[1,4],individu:1,inf:2,infer_valid_mask:4,info:[1,4],inform:[1,2,4],initi:[3,4],inplac:1,input:4,instanc:[1,2,3,4],instanti:[3,4],instead:4,int_to_seq:4,integ:4,integr:1,interest:4,interv:4,invalid:1,invok:3,ipu:4,is_leaf:1,is_root:1,item:4,iter:[1,3],its:4,join:1,just:4,kei:[1,2,4],kernel_s:4,keyword:[1,4],know:4,kwarg:[1,4],l_avg:1,l_drop:1,l_drop_lim:3,l_drop_val:3,l_lo:1,l_up:1,labels_hat:4,last:4,latter:4,lbfg:4,lead:1,leaf:1,learn:[0,1,4],learn_param:4,learner:[1,4],learningratemonitor:4,legend:1,len:4,length:[1,4],less:[1,2],level:[1,3,4],lightn:4,lightningmodul:4,like:4,likelihood:4,lim:3,limit:[1,3],line2d:1,line:1,linear:[1,3],linear_drop:3,linear_drop_uniform:3,linear_uniform:3,lineardrop:[1,3],list:[1,4],litmodel:4,litschedul:4,load:[3,4],load_path:4,log:[1,4],log_dict:4,log_path:1,logger:4,loss:[1,4],loss_func:[1,4],lower:[1,2,3],lr:4,lr_schedul:4,lr_scheduler_config:4,lstm:4,made:4,mai:1,main:4,make:4,make_grid:4,make_model:4,make_q_net:4,make_rng:1,mani:[1,4],map:[1,2,3,4],mask:[1,4],matplotlib:1,max_rollout:[1,2],max_runtim:[1,2],maximum:[1,2],mct:[1,2],mctsnode:1,mdp:[0,1],member:1,membership:1,memori:3,mention:4,metadata:4,method:[1,2,3,4],metric:4,metric_to_track:4,metric_v:4,might:4,mil:0,minimum:1,mixin:1,mlp:4,mlppolici:4,mode:4,model:4,model_cl:4,model_default:4,model_di:4,model_gen:4,model_kwarg:4,modeldefault:4,modul:0,monitor:4,mont:[1,2],more:3,most:4,multi:4,multiextractor:4,multipl:[1,2,4],must:4,myenv:4,n:[1,2,3,4],n_ch:[1,3,4],n_critic:4,n_featur:4,n_gen:[1,3,4],n_gen_learn:1,n_mc:1,n_problem:3,n_rollout:4,n_step:4,n_task:[1,3,4],n_visit:1,name:[1,3,4],namedtupl:3,navi:0,nd:4,ndarrai:[1,2,4],necessari:4,need:4,neg:[1,2,4],net_ch:4,net_task:4,network:4,newlin:4,newshap:1,next:4,nn:4,node:[0,2,4],non:[1,4],none:[1,2,3,4],normal:[1,4],note:[1,2,3],noth:[1,4],np:4,nre:0,nrl:0,num:4,number:[1,2,3,4],numpi:[1,2,3,4],ob:4,object:[1,2,3,4],observ:4,observation_spac:4,often:4,onc:[1,3],one:[1,4],onli:[1,4],openai:[1,4],oper:4,opt_act:4,opt_rollout:4,optim:[1,3,4],optim_cl:4,optim_param:4,optimizer_idx:4,optimizer_on:4,optimizer_step:4,optimizer_two:4,option:[1,2,3,4],order:[2,3,4],org:4,origin:1,os:[1,3,4],otherwis:1,out:[3,4],output:[1,4],over:4,overlap:1,overrid:4,overridden:4,own:[3,4],page:0,panda:1,param:[1,4],param_featur:4,param_gen:3,param_lim:[1,3],param_nam:1,param_prob:3,param_spac:[3,4],paramet:[1,2,3,4],parameter:[1,4],paramref:4,parent:1,partial:[1,2],pass:4,path:[1,3,4],pathlik:[1,3,4],per:[1,4],perform:[0,1,3,4],permut:[1,3,4],permutedtask:3,piecewis:1,piecewiselinear:1,pixel:4,pl:4,place:[1,3],plot:1,plot_lim:1,plot_loss:1,plot_losses_and_schedul:1,plot_schedul:1,plot_task_loss:1,polici:4,pop:4,possibl:4,potenti:1,ppo:4,pre:2,precis:4,predict:4,predict_prob:4,predictor:4,present:4,previou:4,print:[1,2,3,4],priorit:[1,2],prioriti:[1,2],priority_func:[1,2],priority_sort:[1,2],probabl:[3,4],problem:[0,1,4],problem_gen:[1,4],procedur:4,proceed:1,produc:4,program:4,progress:[3,4],project:0,propag:4,properti:[1,3,4],provid:[0,1,4],prune:1,pseudo:1,pseudocod:4,pseudorandom:4,pure:4,put:4,pytorch:4,pytorch_lightn:4,q:4,qnetwork:4,queue:[1,2,3],quickli:4,radar:0,rais:[1,4],random:[1,2,3,4],random_sequenc:2,randomag:4,randomgeneratormixin:[1,3],randomli:[1,3],randomst:[1,2,3,4],rang:4,rate:4,re:[1,2,4],reach:1,recip:4,recommend:4,reducelronplateau:4,referenc:1,reform:[1,4],reform_param_lim:1,reformmixin:1,regist:4,reinforc:[0,1],rel:1,releas:1,render:4,reparam:1,reparameter:1,repeat:3,repeatedli:1,repres:4,represent:[1,4],reproduc:[1,4],requir:[1,4],rescal:4,reset:4,reshap:1,result:0,revers:[1,2],rew:4,reward:4,rgb:4,rgb_arrai:4,rng:[1,2,3,4],roll_out:1,rollout:[1,2],row:4,run:4,runtim:[1,2],s:4,same:[3,4],sampl:[1,4],sample_img:4,save:[3,4],save_path:[3,4],scalar:2,sch:1,schedul:[1,2,3,4],schedulenod:[1,2],schedulenodebound:1,schedulenodereform:1,schedulingproblem:[1,3],schedulingsolut:[1,3],search:[0,1,2],second:4,see:4,seed:[1,2,3,4],select:[1,2],select_child:1,self:[1,3,4],seq:[1,4],seq_append:1,seq_ext:1,seq_extend:1,seq_rem:1,seq_to_int:4,sequenc:[1,2,3,4],sequenti:4,seri:1,set:[1,3,4],sgd:4,shape:[1,4],shift:1,should:4,shown:4,shuffl:3,silent:[1,3,4],similar:4,simpli:4,sinc:4,singl:4,size:4,skip:4,sl:4,slope:1,slope_lim:3,slope_v:3,smooth:4,so:4,solut:[1,2,3],solution_opt:1,solv:[1,3,4],solver:3,some:4,someth:4,sometim:4,sort:[2,4],sort_func:[2,4],sort_wrapp:2,sorted_index:4,sorted_index_inv:4,space:[0,3,4],specif:4,specifi:[1,4],split:3,spork:0,stabl:4,stable_baselines3:4,stablebaselinesschedul:4,stack:[1,3],start:2,state:[1,2,4],statist:1,statu:1,step:4,stop:4,store:[3,4],str:[1,2,3,4],strict:4,string:[1,3,4],stringio:4,structur:4,style:4,subclass:4,submodul:0,subpackag:0,suitabl:4,sum:4,summari:[1,3,4],summarize_task:1,superset:1,supervis:[0,1],support:[3,4],sure:4,t:[1,4],t_drop:1,t_drop_lim:3,t_drop_val:3,t_max:4,t_plot:1,t_releas:1,t_release_lim:3,t_release_v:3,t_run:1,tabulate_kwarg:1,take:4,target:[1,4],task:[2,4],task_gen:3,task_gen_kwarg:3,task_schedul:0,tell:4,tempor:1,tensor:4,termin:4,test:1,text:4,th_visit:[1,2],than:3,them:4,themselv:4,thi:[0,1,2,4],thing:4,those:4,threshold:1,through:4,thu:4,time:[1,2,3,4],to_seri:1,tol:1,toler:1,torch:4,torch_lay:4,torchschedul:4,torchvis:4,total:1,tpu:4,tradit:[0,1],train:[1,4],trainer:4,trainer_kwarg:4,training_step:4,tree:[1,2],truncat:4,truncated_bptt_step:4,tupl:[1,3,4],turn:4,two:4,type:[1,2,3,4],uct:1,under:[0,1],uniform:[1,3],uniformiid:3,uniformli:[3,4],uniqu:4,unit:4,until:1,up:[1,2,4],upcast:1,updat:[1,4],update_stat:1,upper:[1,3],us:[1,2,3,4],user:1,usual:4,util:[0,4],val:4,val_acc:4,val_batch:4,val_data:4,val_loss:4,val_out:4,valid:[1,4],valid_entri:1,validactorcriticpolici:4,validation_epoch_end:4,validation_step:4,validation_step_end:4,validdqnpolici:4,validqnetwork:4,valu:[1,2,3,4],valueerror:1,vari:4,variabl:2,variou:1,verbos:[1,2,3,4],version:3,versu:1,video:4,visit:[1,2],want:4,warn:4,wasserstein:4,we:4,weight:[1,2,4],well:0,what:4,whatev:4,when:[1,4],where:1,wherea:4,whether:4,which:[1,4],whose:4,window:4,within:4,without:4,won:4,wrap:1,wrapper:[0,1],x:[1,4],y:4,yield:[1,3,4],you:4,your:4,z:4},titles:["Task Scheduling package documentation","task_scheduling package","task_scheduling.algorithms package","task_scheduling.generators package","task_scheduling.mdp package"],titleterms:{algorithm:2,base:[1,2,4],channel:3,content:[0,1,2,3,4],document:0,environ:4,featur:4,gener:3,indic:0,mdp:4,modul:[1,2,3,4],node:1,packag:[0,1,2,3,4],problem:3,reinforc:4,result:1,schedul:0,space:1,submodul:[1,2,3,4],subpackag:1,supervis:4,tabl:0,task:[0,1,3],task_schedul:[1,2,3,4],util:1,wrapper:2}})