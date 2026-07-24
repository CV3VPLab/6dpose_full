function summaryPoseDiff(dr, dt)
fprintf( 'angle change(Δr): [%.3f %.3f %.3f] in rotation vector\n', mean(abs(dr)) );
fprintf( 'displacement(Δt): [%.3f %.3f %.3f], %.2f mm\n', mean(abs(dt)), mean(vecnorm(dt, 2, 2)) * 1000 );

