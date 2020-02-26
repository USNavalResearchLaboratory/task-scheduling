function pretty_plot(h)

% Create figure
set(h,'Color',[1 1 1]);

b = get(h,'children');
for q = 1:length(b)
    if strcmp(get(b(q),'type'),'axes')
        set(b(q),'ZColor',[0 0 0],'YColor',[0 0 0],'XColor',[0 0 0],'linewidth',1.0,'fontsize',11, 'Color',[1 1 1],'FontWeight','Bold');
        c = get(gca,'title');
        set(c,'Color',[0 0 0]);
        %          grid on;
        
        c = get(b(q),'Children');
        
        for qq = 1:length(c)
            if strcmp(get(c(qq),'type'),'text')
                set(c(qq),'fontsize',12)
                
            end
            
            
            
        end
        
    end
    if strcmp(get(b(q),'type'),'colorbar')
        set(b(q),'Color',[0 0 0]);
    end
end




cb = findall(h,'type','ColorBar');
if isempty(cb)
    display('No ColorBar.');
end




% CURRENTLY ONLY WORKS WITHOUT SUBPLOTS
if 0
    ax = findobj(h,'Type','Axes');
    NUM_AX = size(ax,1);
    
    for jj = 1:NUM_AX
        
        ax(jj).GridColor = [0 0 0];
        %     aa = ax(jj);
        
        %     ax = ax(1);
        % outerpos = ax.OuterPosition;
        outerpos = [0 0 1 1];
        ti = ax(jj).TightInset;
        if isempty(cb) % No colorbar don't need to account for
            left = outerpos(1) + ti(1);
            bottom = outerpos(2) + ti(2);
            ax_width = outerpos(3) - ti(1) - ti(3);
            ax_height = outerpos(4) - ti(2) - ti(4);
        else
            
            
            cbpos = cb(jj).Position;
            label = cb(jj).Label;
            label.Units = 'normalized';
            gap = cbpos(1) - (ax(jj).Position(1) + ax(jj).Position(3)) ;
            lab_width = label.Extent(1)*cbpos(3);
            
            %     cb_offset = cbpos(1) - outerpos(1) - outerpos(3)
            
            left = outerpos(1) + ti(1);
            bottom = outerpos(2) + ti(2);
            ax_width = outerpos(3) - ti(1) - ti(3) - lab_width - gap;
            %     ax_width = outerpos(3) - ti(1) - ti(3) - cbpos(3) + cb_offset;
            ax_height = outerpos(4) - ti(2) - ti(4);% - cbpos(4);
            
            
        end
        
        
        ax(jj).Position = [left bottom ax_width ax_height];
        
        
    end
    
end