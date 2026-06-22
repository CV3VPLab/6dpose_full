function plot_losses(losses)

if isfield(losses, 'ecc_loss'),  ecc_loss = [losses.ecc_loss]; end
if isfield(losses, 'grad_loss'), grad_loss = [losses.grad_loss]; end
if isfield(losses, 'mask_loss'), mask_loss = [losses.mask_loss]; end
if isfield(losses, 'ssim_loss'), ssim_loss = [losses.ssim_loss]; end
if isfield(losses, 'loss'),      total_loss = [losses.loss]; end

if isfield(losses, 'rgb_loss'),  rgb_loss = [losses.rgb_loss]; end
if isfield(losses, 'blur_loss'), blur_loss = [losses.blur_loss]; end
if isfield(losses, 'fft_loss'),  fft_loss = [losses.fft_loss]; end

iter = [losses.iter];

colors = {"#0072BD", "#D95319", "#EDB120", "#7E2F8E", ...
          "#77AC30", "#4DBEEE", "#A2142F", "#000000"};

figure;
subplot(2,1,1), plot(iter, total_loss, 'Color', colors{8});

subplot(2,1,2), plot(iter, ecc_loss, 'Color', colors{1}, 'LineWidth', 1.5), hold on
subplot(2,1,2), plot(iter, blur_loss, 'Color', colors{2}, 'LineWidth', 1.5);
% % subplot(2,1,2), plot(iter, fft_loss, 'Color', colors{3}, 'LineWidth', 1.5);
subplot(2,1,2), plot(iter, grad_loss, 'Color', colors{4}, 'LineWidth', 1.5);
% subplot(2,1,2), plot(iter, mask_loss, 'Color', colors{5}, 'LineWidth', 1.5);
subplot(2,1,2), plot(iter, ssim_loss, 'Color', colors{6}, 'LineWidth', 1.5);
subplot(2,1,2), plot(iter, rgb_loss, 'Color', colors{7}, 'LineWidth', 1.5);
subplot(2,1,2), legend('ecc', 'blur', 'grad', 'ssim', 'rgb')
hold off

