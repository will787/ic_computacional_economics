%Macroeconomics from the bottom-up  ver. S-D
tic
rand('state',0);                    %reset rand seed

%%%HERE THE INITIALIZATION OF THE MODEL STARTS


%%Parameters
T=100;                              %time
Bk=1;                               %no. of banks
I=100;                              %no. of firms/capitalists
J=650;                              %no. of workers
div=0.2;                            %share of dividends
refi=0.02;                          %general refinancing rate
alpha=0.1;                          %labour productivity
phi=0.8;                            %credit demand contraction
tau=0.05;                           %rate of debt reimbursment
wb=1;                               %wage rate
z=2;                                %no. of aplications in consumption good market
c=0.8;                              %propensity to consume

%%Initial conditions

%firms
A=10*ones(1,I);                     %firm equity
P=ones(1,I);                        %prices
Y=zeros(1,I);                       %current production/stocks
Y_prev=ones(1,I);                   %past production
B=zeros(1,I);                       %current credit demand
Q=zeros(1,I);                       %sales   
proposed_rate=zeros(1,I);           %bid interest rates
interests=zeros(1,I);               %interests
Leff=zeros(1,I);                    %current employees
pi=zeros(1,I);                      %firm profits
De=ones(1,I);                       %expected demand
vacancies=zeros(1,I);               %job vacancies  
deb=zeros(1,I);                     %firm debts
liquidity=A;                        %firm liquid resources

%households
w=zeros(1,J);                       %earned wages
PA=zeros(1,J+I);                    %household personal assets (saving stock)
Oc=zeros(1,J);                      %employment status: Oc(i)=j --> worker j employed by firm i; if i=0, j is unemployed

%bank
E=ones(1,Bk)*500;                   %bank's equity
loans=0;                            %total loans

%macro time series
Ytot=zeros(1,T);                    %total production
consumption=zeros(1,T);             %total cunsumption  
cum_wages=zeros(1,T);               %total wages
wealthF=zeros(1,T);                 %total firm liquidity
wealthHH=zeros(1,T);                %total household savings
infla_rate=0;                       %inflation rate
price=[1];                          %consumer price index time series
Un=zeros(1,T);                      %unemployment
interestrate=zeros(1,T);            %average interest rate
leverage_ratio=zeros(1,T);          
dividends=zeros(1,T);               %total dividends
baddebt=zeros(1,T);                 %total bad debts
totE=zeros(1,T);                    %bank equity time series
credits=zeros(1,T);                 %total credit lent in each period
Ftot=zeros(1,I);                    %borrowings
debits=zeros(1,T);                  %total stock of debts
defaults=zeros(1,T);                %number of bankruptcies
profits=zeros(T,I);                 %panel of firm profits
profitsB=zeros(T,Bk);               %bank profit time series


%%%END OF MODEL INITIALIZATION



%%%HEREAFTER THE MODEL STARTS RUNNING FOR T PERIODS


for t=1:T                                                               %repeat for T times the above instructions
    
    if A<=0                                                             %if all the firms are defaulted, then stop simulation 
        break
    end
    
    %firms define expected demands and bid prices
    for i=1:I
        if Y(i)==0 & P(i)>=price(t)
            De(i)=Y_prev(i)*(1+rand*0.1);
        elseif Y(i)==0 & P(i)<price(t)
            De(i)=Y_prev(i);
            P(i)=P(i)*(1+rand*0.1);
        elseif Y(i)>0 & P(i)>price(t)
            De(i)=Y_prev(i); 
            P(i)=P(i)*(1-rand*0.1);
        elseif Y(i)>0 & P(i)<=price(t)
            De(i)=Y_prev(i)*(1-rand*0.1);
        end    
        
        if De(i)<alpha                                                  %in order to hire at least one worker
            De(i)=alpha;    
        end
    end
    
    
    %labour demand (stock) 
    Ld=ceil(De./alpha);                                                 %labour demand approximated by excess
    wages=wb*Ld;   
    
    
%%CREDIT MARKET OPENS

        
    %bank decides credit conditions
    Ftot(:)=0; 
    proposed_rate(:)=0;

    B=wages-liquidity;                                                  %credit demand (financial needs)          
    d=(deb+B)./(liquidity+0.01);                                        %vector of leverage ratios 
    loan_applications=find(B>0);                                        %only firms with positive credit demand will apply for an additional loan
    proposed_rate=refi*(1+rand)*(1+tanh(d));                            %the bank charges the interest rates for each firm
    maxrate=0.05+infla_rate;                                            %interest rate threshold
    
    %firms choose how much to borrow    
    for i=loan_applications
        if proposed_rate(i)<=maxrate  
            credit=B(i);    
        else credit=phi*B(i);
        end
        deb(i)=deb(i)+credit;                                           %update firm's debt stock
        loans=loans+credit;                                             %update bank's credit stock
        Ftot(i)=credit;                                                 %record flow of new credit for firm i
    end       
    
    
%%CREDIT MARKET CLOSES    
    

    %re-define labour demand
    Ld=min(Ld,(Ftot+liquidity)/wb);                                     %demand (stock)     
    Ld=floor(Ld);                                                       %round to the lower integer, otherwise not enough money to pay wages
    vacancies=Ld-Leff;                                                  %definitive labour demand (flow)    
    
    
%%JOB MARKET OPENS


    surplus=find(vacancies<0);                                          %firms with too many workers
    deficit=find(vacancies>0);                                          %firms with too few workers
    
    %firms with excess workforce fire
    for i=surplus   
        workforce=find(Oc==i);                                          %pick all firm i's workers
        f=randperm(length(workforce));
        f=f(1:-vacancies(i));                                           %take randomly "-vacancies(i)" workers and fire them
        fired=workforce(f);  
        Oc(fired)=0;
        w(fired)=0;
        Leff(i)=Ld(i);                                                  %update no. of workers
    end
    
    %firms with insufficient workforce hire
    for q=randperm(length(deficit))                                     %pick firms randomly, so that no one is privileged
        i=deficit(q);  
        unemployed=find(Oc==0);                                         %spot currently unemployed workers
        empl=min(vacancies(i),length(unemployed));
        e=randperm(length(unemployed));                                 %choose randomly among unemployed workers
        e=e(1:empl);    
        hired=unemployed(e);                                            %hire "e" unemployed people
        Oc(hired)=i;
        w(hired)=wb;
        Leff(i)=Leff(i)+empl;                                           %update no. of workers
    end  
    
    
%%JOB MARKET CLOSES
    
    
    %production
    Y=min(De,alpha*Leff);    
    Y_prev=Y;
    
    %compute interests
    interests=proposed_rate.*Ftot;    
    
    %total wages paid by firms
    wages=wb*Leff;
    cum_wages(t)=sum(wages);
    
    %minimum prices
    for i=1:I
        
        if Y(i)>0
            Pl(i)=1.05*(wages(i)+interests(i))/Y(i);                    %break-even price
        else Pl(i)=0;
        end
        
        if P(i)<Pl(i)
            P(i)=Pl(i);
        end
        
    end
        

%%CONSUMPTION GOOD MARKET OPENS
    
        
    %workers receive wages
    PA(1:J)=PA(1:J)+w;    
    
    %consumers compute their consumption budgets
    cons_budget=c*PA;
    PA=PA-cons_budget;        
    consumers=find(cons_budget>0);
    
    %search and matching starts
    Q(:)=0;
    
    for wor=randperm(length(consumers))      
        j=consumers(wor);                                               %randomly pick a consumer
        list=randperm(I);
        Z=list(1:z);                                                    %create the individual market
        PZ=P(Z);  
        [PZ,order]=sort(PZ);                                            %sort prices in ascending order
        flag=1;
        
        while (cons_budget(j)>0 & flag<=z)                              %continue buying till budget is positive and there are firms available
            best=Z(order(flag));                                        %pick first best firm; with flag increasing, pick the second best, the third...   
            
            if Y(best)>0                                                %buy if 'best' firm has still positive stocks 
                p=P(best);
                if Y(best) > cons_budget(j)/p           
                    Y(best)=Y(best)-cons_budget(j)/p;                   %reduce stocks
                    Q(best)=Q(best)+cons_budget(j)/p;                   %update sales
                    consumption(t)=consumption(t)+cons_budget(j)/p;     
                    cons_budget(j)=0;                                   %j spends all its budget  
                elseif  Y(best) <= cons_budget(j)/p  
                    cons_budget(j)=cons_budget(j)- Y(best)*p;   
                    Q(best)=Q(best)+Y(best);    
                    consumption(t)=consumption(t)+Y(best);
                    Y(best)=0;    
                end    
            end
            flag=flag+1;                                                %increase counter
        end 
        
    end    
    
    %search and matching ends
    
    %unvoluntary savings are added to the vuluntary ones
    PA=PA+cons_budget;
    
    
%%CONSUMPTION GOOD MARKET CLOSES


%%ACCOUNTING
    
    %firm revenues
    RIC=P.*Q;   
    liquidity=liquidity+RIC+Ftot-wages-interests-tau*deb;
    loans=loans-sum(deb)*tau;
    deb=(1-tau)*deb;
    
    %firm profits
    pi=RIC-wages-interests;                                             %final profits
    profits(t,:)=pi;
    
    %dividends
    pospi=find(pi>0);                                                   %pick firms with positive profits
    for i=pospi
        divi=div*pi(i);                                                 %compute dividends paid by firm i
        PA(J+i)=PA(J+i)+divi;                                           %dividends paid to firm i owner
        liquidity(i)=liquidity(i)-divi;
        dividends(t)=dividends(t)+divi;
        pi(i)=pi(i)-divi;
    end
    
    %update firms'equity
    A=A+pi;                                                             
    
    %replacement of bankrupted firms 
    piB=0;                                                              %reset bank's profits
    Dep=De(A>0);                                                        %take 3 control variables of active firms
    Pp=P(A>0);
    Y_prevp=Y_prev(A>0);  
    
    negcash=find(liquidity<=0);                                         %find defaulted firms
    
    for i=negcash                                                       %pick sequentially failed firms
        defaults(t)=defaults(t)+1;
        zzz=deb(i);                                                     %take residual debts
        if zzz>0
            piB=piB+A(i);                                               %account for bad debts
            baddebt(t)=baddebt(t)+zzz/(1-tau);
            loans=loans-zzz;
        end
        A(i)=PA(J+i);                                                   %initialize new firm 
        PA(J+i)=0;
        liquidity(i)=A(i);
        deb(i)=0;
        De(i)=trimmean(Dep,10);
        P(i)=trimmean(Pp,10);
        Y_prev(i)=trimmean(Y_prevp,10);
        Y(i)=1;
    end
    
    %bank accounting
    piB=piB+sum(interests);                                             %bank profits  
    E=E+piB;                                                            %update bank capital
    profitsB(t)=piB;
    
    %inflation rate
    if sum(Q)>0 
        share=(Q)/sum(Q);
    else share=ones(1,I)/I;
    end
    RPI=P*share';                                                       %retail price index
    infla_rate=RPI/price(t)-1;    
    price=[price RPI];    
    
    %unemployment rate
    disocct=(J-length(Oc(Oc>0)))/J;
    Un(t)=disocct;
    
    %average interest rate
    if sum(Ftot)==0
        interestrate(t)=0;
    else
        interestrate(t)=sum(proposed_rate.*Ftot)/sum(Ftot);  
    end
    
    %other state variables
    Ytot(t)=sum(Y_prev);                                                %total production
    credits(t)=sum(Ftot);
    debits(t)=loans;
    wealthHH(t)=sum(PA);
    wealthF(t)=sum(liquidity);
    totE(t)=E;
    leverage_ratio(t)=sum(deb)/sum(liquidity);

    
%%ACCOUNTING ENDS


end   


%%%END OF SIMULATION
toc
