function plot_losses(losses)

fLosses = false(8, 1);
if hasVar(losses, 'loss'),      fLosses(1) = true; total_loss = [losses.loss]; end
if hasVar(losses, 'ecc_loss'),  fLosses(2) = true; ecc_loss = [losses.ecc_loss]; end
if hasVar(losses, 'ssim_loss'), fLosses(3) = true; ssim_loss = [losses.ssim_loss]; end
if hasVar(losses, 'grad_loss'), fLosses(4) = true; grad_loss = [losses.grad_loss]; end
if hasVar(losses, 'blur_loss'), fLosses(5) = true; blur_loss = [losses.blur_loss]; end
if hasVar(losses, 'mask_loss'), fLosses(6) = true; mask_loss = [losses.mask_loss]; end
if hasVar(losses, 'rgb_loss'),  fLosses(7) = true; rgb_loss = [losses.rgb_loss]; end
if hasVar(losses, 'track_loss'),  fLosses(8) = true; track_loss = [losses.track_loss]; end

iter = [losses.iter] + 1;

colors = {"#0072BD", "#D95319", "#EDB120", "#7E2F8E", ...
          "#77AC30", "#4DBEEE", "#A2142F", "#000000"};

figure; set(gcf, 'Color', 'w')

subplot(3,1,1), plot(iter, total_loss, 'Color', colors{8}), hold on
set(gca, 'Color', 'w')
set(gca, 'XColor', 'k')
set(gca, 'YColor', 'k')
labels = {'loss'};
if fLosses(8)
    subplot(3,1,1), plot(iter, track_loss, 'Color', colors{1}, 'LineWidth', 1.5);
    labels{end+1} = 'track';
end
subplot(3,1,1), legend(labels)
hold off

%
subplot(3,1,2), plot(iter, ecc_loss, 'Color', colors{1}, 'LineWidth', 1.5), hold on
set(gca, 'Color', 'w')
set(gca, 'XColor', 'k')
set(gca, 'YColor', 'k')
labels = {'ecc'};

if fLosses(3)
    subplot(3,1,2), plot(iter, ssim_loss, 'Color', colors{6}, 'LineWidth', 1.5);
    labels{end+1} = 'ssim';
end
if fLosses(4)
    subplot(3,1,2), plot(iter, grad_loss, 'Color', colors{4}, 'LineWidth', 1.5);
    labels{end+1} = 'grad';
end
if fLosses(5)
    subplot(3,1,2), plot(iter, blur_loss, 'Color', colors{2}, 'LineWidth', 1.5);
    labels{end+1} = 'blur';
end
if fLosses(6)
    subplot(3,1,2), plot(iter, mask_loss, 'Color', colors{5}, 'LineWidth', 1.5);
    labels{end+1} = 'mask';
end
if fLosses(7)
    subplot(3,1,2), plot(iter, rgb_loss, 'Color', colors{7}, 'LineWidth', 1.5);
    labels{end+1} = 'rgb';
end

subplot(3,1,2), legend(labels)
hold off

if fLosses(3)
    subplot(3,1,3), plot(iter, ssim_loss, 'Color', colors{6}, 'LineWidth', 1.5); hold on
    set(gca, 'Color', 'w')
    set(gca, 'XColor', 'k')
    set(gca, 'YColor', 'k')
    labels{end+1} = 'ssim';
end
if fLosses(4)
    subplot(3,1,3), plot(iter, grad_loss, 'Color', colors{4}, 'LineWidth', 1.5);
    labels{end+1} = 'grad';
end
if fLosses(5)
    subplot(3,1,3), plot(iter, blur_loss, 'Color', colors{2}, 'LineWidth', 1.5);
    labels{end+1} = 'blur';
end
if fLosses(6)
    subplot(3,1,3), plot(iter, mask_loss, 'Color', colors{5}, 'LineWidth', 1.5);
    labels{end+1} = 'mask';
end
if fLosses(7)
    subplot(3,1,3), plot(iter, rgb_loss, 'Color', colors{7}, 'LineWidth', 1.5);
    labels{end+1} = 'rgb';
end
subplot(3,1,3), legend(labels(2:end))
hold off

end

function bExist = hasVar(T, name)    
    bExist = any(strcmp(name, string(T.Properties.VariableNames)));
end

