## -----------------------------------------------------------------------------
library(spStack)
library(ggplot2)
set.seed(1729)

## -----------------------------------------------------------------------------
# training and test data sizes
n_train <- 150
n_pred <- 50

data("simGaussian")
dat_train <- simGaussian[1:n_train, ]
dat_pred <- simGaussian[n_train + 1:n_pred, ]

mod1 <- spLMstack(y ~ x1, data = dat_train,
                  coords = as.matrix(dat_train[, c("s1", "s2")]),
                  cor.fn = "matern",
                  params.list = list(phi = c(1.5, 3, 5),
                                     nu = c(0.75, 1.25),
                                     noise_sp_ratio = c(0.5, 1, 2)),
                  n.samples = 1000, loopd.method = "psis",
                  parallel = FALSE, solver = "ECOS", verbose = TRUE)

## -----------------------------------------------------------------------------
sp_pred <- as.matrix(dat_pred[, c("s1", "s2")])
X_new <- as.matrix(cbind(rep(1, n_pred), dat_pred$x1))
mod.pred <- posteriorPredict(mod1, coords_new = sp_pred, covars_new = X_new, joint = TRUE)
post_samps <- stackedSampler(mod.pred)

## ----fig.align='center', fig.height=3.5, fig.width=7--------------------------
postpred_z <- post_samps$z.pred
post_z_summ <- t(apply(postpred_z, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
z_combn <- data.frame(z = dat_pred$z_true, zL = post_z_summ[, 1],
                      zM = post_z_summ[, 2], zU = post_z_summ[, 3])
plot_z_summ <- ggplot(data = z_combn, aes(x = z)) +
  geom_errorbar(aes(ymin = zL, ymax = zU), alpha = 0.5, color = "skyblue") +
  geom_point(aes(y = zM), size = 0.5, color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True z1") + ylab("Posterior of z1") + theme_bw() +
  theme(panel.grid = element_blank(), aspect.ratio = 1)

postpred_y <- post_samps$y.pred
post_y_summ <- t(apply(postpred_y, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
y_combn <- data.frame(y = dat_pred$y, yL = post_y_summ[, 1],
                      yM = post_y_summ[, 2], yU = post_y_summ[, 3])

plot_y_summ <- ggplot(data = y_combn, aes(x = y)) +
  geom_errorbar(aes(ymin = yL, ymax = yU), alpha = 0.5, color = "skyblue") +
  geom_point(aes(y = yM), size = 0.5, color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True y") + ylab("Posterior of y") + theme_bw() +
  theme(panel.grid = element_blank(), aspect.ratio = 1)

ggpubr::ggarrange(plot_z_summ, plot_y_summ)

## -----------------------------------------------------------------------------
# training and test data sizes
n_train <- 150
n_pred <- 50

# load spatial Poisson data
data("simPoisson")
dat_train <- simPoisson[1:n_train, ]
dat_pred <- simPoisson[n_train + 1:n_pred, ]

mod1 <- spGLMstack(y ~ x1, data = dat_train, family = "poisson",
                   coords = as.matrix(dat_train[, c("s1", "s2")]), cor.fn = "matern",
                   params.list = list(phi = c(3, 4, 5), nu = c(0.5, 1.0),
                                      boundary = c(0.5)),
                   priors = list(nu.beta = 5, nu.z = 5),
                   n.samples = 1000,
                   loopd.controls = list(method = "CV", CV.K = 10, nMC = 500),
                   verbose = TRUE)

## -----------------------------------------------------------------------------
sp_pred <- as.matrix(dat_pred[, c("s1", "s2")])
X_new <- as.matrix(cbind(rep(1, n_pred), dat_pred$x1))
mod.pred <- posteriorPredict(mod1, coords_new = sp_pred, covars_new = X_new, joint = FALSE)
post_samps <- stackedSampler(mod.pred)

## ----fig.align='center', fig.height=3.5, fig.width=7--------------------------
postpred_z <- post_samps$z.pred
post_z_summ <- t(apply(postpred_z, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
z_combn <- data.frame(z = dat_pred$z_true, zL = post_z_summ[, 1],
                      zM = post_z_summ[, 2], zU = post_z_summ[, 3])

plot_z_summ <- ggplot(data = z_combn, aes(x = z)) +
  geom_errorbar(aes(ymin = zL, ymax = zU), alpha = 0.5, color = "skyblue") +
  geom_point(aes(y = zM), size = 0.5, color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True z") + ylab("Posterior predictive of z") + theme_bw() +
  theme(panel.grid = element_blank(), aspect.ratio = 1)

postpred_y <- post_samps$y.pred
post_y_summ <- t(apply(postpred_y, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
y_combn <- data.frame(y = dat_pred$y, yL = post_y_summ[, 1],
                      yM = post_y_summ[, 2], yU = post_y_summ[, 3])

plot_y_summ <- ggplot(data = y_combn, aes(x = y)) +
  geom_errorbar(aes(ymin = yL, ymax = yU), alpha = 0.5, color = "skyblue") +
  geom_point(aes(y = yM), size = 0.5, color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True y") + ylab("Posterior predictive of y") + theme_bw() +
  theme(panel.grid = element_blank(), aspect.ratio = 1)

ggpubr::ggarrange(plot_z_summ, plot_y_summ)

## -----------------------------------------------------------------------------
# Example 2: Spatial-temporal model with varying coefficients
n_train <- 150
n_pred <- 50
data("sim_stvcPoisson")
dat <- sim_stvcPoisson[1:(n_train + n_pred), ]

# split dataset into test and train
dat_train <- dat[1:n_train, ]
dat_pred <- dat[n_train + 1:n_pred, ]

# create list of candidate models (multivariate)
mod.list2 <- candidateModels(list(phi_s = list(1, 2, 3),
                                  phi_t = list(1, 2, 4),
                                  boundary = c(0.5, 0.75)), "cartesian")

# fit a spatial-temporal varying coefficient model using predictive stacking
mod1 <- stvcGLMstack(y ~ x1 + (x1), data = dat_train, family = "poisson",
                     sp_coords = as.matrix(dat_train[, c("s1", "s2")]),
                     time_coords = as.matrix(dat_train[, "t_coords"]),
                     cor.fn = "gneiting-decay",
                     process.type = "multivariate",
                     candidate.models = mod.list2,
                     loopd.controls = list(method = "CV", CV.K = 10, nMC = 500),
                     n.samples = 500)

## -----------------------------------------------------------------------------
# prepare new coordinates and covariates for prediction
sp_pred <- as.matrix(dat_pred[, c("s1", "s2")])
tm_pred <- as.matrix(dat_pred[, "t_coords"])
X_new <- as.matrix(cbind(rep(1, n_pred), dat_pred$x1))
mod_pred <- posteriorPredict(mod1,
                             coords_new = list(sp = sp_pred, time = tm_pred),
                             covars_new = list(fixed = X_new, vc = X_new),
                             joint = FALSE)

# sample from the stacked posterior and posterior predictive distribution
post_samps <- stackedSampler(mod_pred)

## ----fig.align='center', fig.height=3.5, fig.width=7--------------------------
postpred_z <- post_samps$z.pred
post_z1_summ <- t(apply(postpred_z[1:n_pred,], 1,
                        function(x) quantile(x, c(0.025, 0.5, 0.975))))
post_z2_summ <- t(apply(postpred_z[n_pred + 1:n_pred,], 1,
                        function(x) quantile(x, c(0.025, 0.5, 0.975))))

z1_combn <- data.frame(z = dat_pred$z1_true, zL = post_z1_summ[, 1],
                       zM = post_z1_summ[, 2], zU = post_z1_summ[, 3])
z2_combn <- data.frame(z = dat_pred$z2_true, zL = post_z2_summ[, 1],
                       zM = post_z2_summ[, 2], zU = post_z2_summ[, 3])

library(ggplot2)
plot_z1_summ <- ggplot(data = z1_combn, aes(x = z)) +
  geom_errorbar(aes(ymin = zL, ymax = zU), alpha = 0.5, color = "skyblue") +
  geom_point(aes(y = zM), size = 0.5, color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True z1") + ylab("Posterior predictive of z1") + theme_bw() +
  theme(panel.grid = element_blank(), aspect.ratio = 1)

plot_z2_summ <- ggplot(data = z2_combn, aes(x = z)) +
  geom_errorbar(aes(ymin = zL, ymax = zU), alpha = 0.5, color = "skyblue") +
  geom_point(aes(y = zM), size = 0.5, color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True z2") + ylab("Posterior predictive of z2") + theme_bw() +
  theme(panel.grid = element_blank(), aspect.ratio = 1)

ggpubr::ggarrange(plot_z1_summ, plot_z2_summ)

