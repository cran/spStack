## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
set.seed(1729)

## ----setup--------------------------------------------------------------------
library(spStack)

## -----------------------------------------------------------------------------
data("simGaussian")
dat <- simGaussian[1:200, ] # work with first 200 rows

muBeta <- c(0, 0)
VBeta <- cbind(c(1E4, 0.0), c(0.0, 1E4))
sigmaSqIGa <- 2
sigmaSqIGb <- 2
phi0 <- 3
nu0 <- 0.5
noise_sp_ratio <- 0.8
prior_list <- list(beta.norm = list(muBeta, VBeta),
                   sigma.sq.ig = c(sigmaSqIGa, sigmaSqIGb))
nSamples <- 1000

## ----spLMexactLOO_exact-------------------------------------------------------
mod1 <- spLMexact(y ~ x1, data = dat,
                  coords = as.matrix(dat[, c("s1", "s2")]),
                  cor.fn = "matern",
                  priors = prior_list,
                  spParams = list(phi = phi0, nu = nu0),
                  noise_sp_ratio = noise_sp_ratio, n.samples = nSamples,
                  loopd = TRUE, loopd.method = "exact",
                  verbose = TRUE)

## -----------------------------------------------------------------------------
post_beta <- mod1$samples$beta
summary_beta <- t(apply(post_beta, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
rownames(summary_beta) <- mod1$X.names
print(summary_beta)

## ----spLMexactLOO_PSIS--------------------------------------------------------
mod2 <- spLMexact(y ~ x1, data = dat,
                  coords = as.matrix(dat[, c("s1", "s2")]),
                  cor.fn = "matern",
                  priors = prior_list,
                  spParams = list(phi = phi0, nu = nu0),
                  noise_sp_ratio = noise_sp_ratio, n.samples = nSamples,
                  loopd = TRUE, loopd.method = "PSIS",
                  verbose = FALSE)

## ----fig.align='center'-------------------------------------------------------
loopd_exact <- mod1$loopd
loopd_psis <- mod2$loopd
loopd_df <- data.frame(exact = loopd_exact, psis = loopd_psis)

library(ggplot2)
plot1 <- ggplot(data = loopd_df, aes(x = exact)) +
  geom_point(aes(y = psis), size = 1, alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", alpha = 0.5) +
  xlab("Exact") + ylab("PSIS") + theme_bw() +
  theme(panel.background = element_blank(), 
        panel.grid = element_blank(), aspect.ratio = 1)
plot1

## ----spLMstack----------------------------------------------------------------
mod3 <- spLMstack(y ~ x1, data = dat,
                  coords = as.matrix(dat[, c("s1", "s2")]),
                  cor.fn = "matern",
                  priors = prior_list,
                  params.list = list(phi = c(1.5, 3, 5),
                                     nu = c(0.5, 1, 1.5),
                                     noise_sp_ratio = c(0.5, 1.5)),
                  n.samples = 1000, loopd.method = "exact",
                  parallel = FALSE, solver = "ECOS", verbose = TRUE)

## -----------------------------------------------------------------------------
print(mod3$solver.status)
print(mod3$run.time)

## -----------------------------------------------------------------------------
post_samps <- stackedSampler(mod3)

## -----------------------------------------------------------------------------
post_beta <- post_samps$beta
summary_beta <- t(apply(post_beta, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
rownames(summary_beta) <- mod3$X.names
print(summary_beta)

## ----fig.align='center', fig.height=3.5, fig.width=7, fig.alt="Posterior distributions of the fixed effects"----
library(tidyr)
library(dplyr)

post_beta_df <- as.data.frame(post_beta)
post_beta_df <- post_beta_df %>%
  mutate(row = paste0("beta", row_number()-1)) %>%
  pivot_longer(-row, names_to = "sample", values_to = "value")

# True values of beta0 and beta1
truth <- data.frame(row = c("beta0", "beta1"), true_value = c(2, 5))

ggplot(post_beta_df, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.6) +
  geom_vline(data = truth, aes(xintercept = true_value), 
             color = "red", linetype = "dashed", linewidth = 0.5) +
  facet_wrap(~ row, scales = "free") + labs(x = "", y = "Density") +
  theme_bw() + theme(panel.background = element_blank(), 
                     panel.grid = element_blank(), aspect.ratio = 1)

## ----fig.align='center', fig.alt="Comparison of stacked posterior with the true values"----
post_z <- post_samps$z
post_z_summ <- t(apply(post_z, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
z_combn <- data.frame(z = dat$z_true, zL = post_z_summ[, 1],
                      zM = post_z_summ[, 2], zU = post_z_summ[, 3])

plotz <- ggplot(data = z_combn, aes(x = z)) +
  geom_point(aes(y = zM), size = 0.75, color = "darkblue", alpha = 0.5) +
  geom_errorbar(aes(ymin = zL, ymax = zU), width = 0.05, alpha = 0.15,
                color = "skyblue") +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  xlab("True z") + ylab("Stacked posterior of z") + theme_bw() +
  theme(panel.background = element_blank(), 
        panel.grid = element_blank(), aspect.ratio = 1)
plotz

## ----fig.align='center', fig.height=3.5, fig.width=7, fig.alt="Comparison of the interploated spatial surfaces of the true random effects and the posterior medians."----
postmedian_z <- apply(post_z, 1, median)
dat$z_hat <- postmedian_z
plot_z <- surfaceplot2(dat, coords_name = c("s1", "s2"),
                       var1_name = "z_true", var2_name = "z_hat")
library(ggpubr)
ggarrange(plotlist = plot_z, common.legend = TRUE, legend = "right")

## ----fig.align='center'-------------------------------------------------------
data("simPoisson")
dat <- simPoisson[1:200, ] # work with first 200 observations

ggplot(dat, aes(x = s1, y = s2)) +
  geom_point(aes(color = y), alpha = 0.75) +
  scale_color_distiller(palette = "RdYlGn", direction = -1,
                        label = function(x) sprintf("%.0f", x)) +
  guides(alpha = 'none') + theme_bw() +
  theme(axis.ticks = element_line(linewidth = 0.25),
        panel.background = element_blank(), panel.grid = element_blank(),
        legend.title = element_text(size = 10, hjust = 0.25),
        legend.box.just = "center", aspect.ratio = 1)

## ----spGLMexact_Pois----------------------------------------------------------
mod1 <- spGLMexact(y ~ x1, data = dat, family = "poisson",
                   coords = as.matrix(dat[, c("s1", "s2")]), cor.fn = "matern",
                   spParams = list(phi = phi0, nu = nu0),
                   priors = list(nu.beta = 5, nu.z = 5),
                   boundary = 0.5,
                   n.samples = 1000, verbose = TRUE)

## -----------------------------------------------------------------------------
post_beta <- mod1$samples$beta
summary_beta <- t(apply(post_beta, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
rownames(summary_beta) <- mod1$X.names
print(summary_beta)

## -----------------------------------------------------------------------------
mod1 <- recoverGLMscale(mod1)

## ----fig.align='center', fig.height=3.5, fig.width=7, fig.alt="Posterior distributions of the scale parameters of the fixed and the random effects."----
post_scale_df <- data.frame(value = sqrt(c(mod1$samples$sigmasq.beta, mod1$samples$sigmasq.z)),
                            group = factor(rep(c("sigma.beta", "sigma.z"), 
                                    each = length(mod1$samples$sigmasq.beta))))
ggplot(post_scale_df, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.6) +
  facet_wrap(~ group, scales = "free") + labs(x = "", y = "Density") +
  theme_bw() + theme(panel.background = element_blank(), 
                     panel.grid = element_blank(), aspect.ratio = 1)

## ----spGLMstack_Pois----------------------------------------------------------
mod2 <- spGLMstack(y ~ x1, data = dat, family = "poisson",
                   coords = as.matrix(dat[, c("s1", "s2")]), cor.fn = "matern",
                   params.list = list(phi = c(3, 7, 10), nu = c(0.5, 1.5),
                                      boundary = c(0.5, 0.6)),
                   n.samples = 1000, priors = list(mu.beta = 5, nu.z = 5),
                   loopd.controls = list(method = "CV", CV.K = 10, nMC = 1000),
                   parallel = TRUE, solver = "ECOS", verbose = TRUE)

## -----------------------------------------------------------------------------
print(mod2$solver.status)
print(mod2$run.time)

## -----------------------------------------------------------------------------
mod2 <- recoverGLMscale(mod2)

## -----------------------------------------------------------------------------
post_samps <- stackedSampler(mod2)

## -----------------------------------------------------------------------------
post_beta <- post_samps$beta
summary_beta <- t(apply(post_beta, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
rownames(summary_beta) <- mod3$X.names
print(summary_beta)

## ----fig.align='center', fig.height=3.5, fig.width=7, fig.alt="Posterior distributions of the fixed effects"----
post_beta_df <- as.data.frame(post_beta)
post_beta_df <- post_beta_df %>%
  mutate(row = paste0("beta", row_number()-1)) %>%
  pivot_longer(-row, names_to = "sample", values_to = "value")

# True values of beta0 and beta1
truth <- data.frame(row = c("beta0", "beta1"), true_value = c(2, -0.5))

ggplot(post_beta_df, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.6) +
  geom_vline(data = truth, aes(xintercept = true_value), 
             color = "red", linetype = "dashed", linewidth = 0.5) +
  facet_wrap(~ row, scales = "free") + labs(x = "", y = "Density") +
  theme_bw() + theme(panel.background = element_blank(), 
                     panel.grid = element_blank(), aspect.ratio = 1)

## ----fig.align='center'-------------------------------------------------------
post_z <- post_samps$z
post_z_summ <- t(apply(post_z, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
z_combn <- data.frame(z = dat$z_true, zL = post_z_summ[, 1],
                      zM = post_z_summ[, 2], zU = post_z_summ[, 3])

plotz <- ggplot(data = z_combn, aes(x = z)) +
  geom_point(aes(y = zM), size = 0.75, color = "darkblue", alpha = 0.5) +
  geom_errorbar(aes(ymin = zL, ymax = zU), width = 0.05, alpha = 0.15,
                color = "skyblue") +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  xlab("True z") + ylab("Stacked posterior of z") + theme_bw() +
  theme(panel.background = element_blank(), 
        panel.grid = element_blank(), aspect.ratio = 1)
plotz

## ----fig.align='center', fig.height=3.5, fig.width=7--------------------------
postmedian_z <- apply(post_z, 1, median)
dat$z_hat <- postmedian_z
plot_z <- surfaceplot2(dat, coords_name = c("s1", "s2"),
                       var1_name = "z_true", var2_name = "z_hat")
library(ggpubr)
ggarrange(plotlist = plot_z, common.legend = TRUE, legend = "right")

## -----------------------------------------------------------------------------
data("simBinom")
dat <- simBinom[1:200, ] # work with first 200 rows

mod1 <- spGLMexact(cbind(y, n_trials) ~ x1, data = dat, family = "binomial",
                   coords = as.matrix(dat[, c("s1", "s2")]), cor.fn = "matern",
                   spParams = list(phi = 3, nu = 0.5),
                   boundary = 0.5, n.samples = 1000, verbose = FALSE)

## -----------------------------------------------------------------------------
post_beta <- mod1$samples$beta
summary_beta <- t(apply(post_beta, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
rownames(summary_beta) <- mod1$X.names
print(summary_beta)

## -----------------------------------------------------------------------------
data("simBinary")
dat <- simBinary[1:200, ]

mod1 <- spGLMexact(y ~ x1, data = dat, family = "binary",
                   coords = as.matrix(dat[, c("s1", "s2")]), cor.fn = "matern",
                   spParams = list(phi = 4, nu = 0.4),
                   boundary = 0.5, n.samples = 1000, verbose = FALSE)

## -----------------------------------------------------------------------------
post_beta <- mod1$samples$beta
summary_beta <- t(apply(post_beta, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
rownames(summary_beta) <- mod1$X.names
print(summary_beta)

