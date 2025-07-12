## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
set.seed(1729)

## -----------------------------------------------------------------------------
library(spStack)

# training and test data sizes
n_train <- 100
n_pred <- 50

data("simGaussian")
dat_train <- simGaussian[1:n_train, ]
dat_pred <- simGaussian[n_train + 1:n_pred, ]

## -----------------------------------------------------------------------------
mod1 <- spLMstack(y ~ x1, data = dat_train,
                  coords = as.matrix(dat_train[, c("s1", "s2")]),
                  cor.fn = "matern",
                  params.list = list(phi = c(1.5, 3, 5),
                                     nu = c(0.75, 1.25),
                                     noise_sp_ratio = c(0.5, 1, 2)),
                  n.samples = 1000, loopd.method = "psis",
                  parallel = FALSE, solver = "ECOS", verbose = TRUE)

## -----------------------------------------------------------------------------
post_samps <- stackedSampler(mod1)

## -----------------------------------------------------------------------------
post_beta <- post_samps$beta
summary_beta <- t(apply(post_beta, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
rownames(summary_beta) <- mod1$X.names
print(summary_beta)

## -----------------------------------------------------------------------------
sp_pred <- as.matrix(dat_pred[, c("s1", "s2")])
X_new <- as.matrix(cbind(rep(1, n_pred), dat_pred$x1))

## -----------------------------------------------------------------------------
mod.pred <- posteriorPredict(mod1,
                             coords_new = sp_pred,
                             covars_new = X_new,
                             joint = TRUE)

## -----------------------------------------------------------------------------
postpred_samps <- stackedSampler(mod.pred)

## -----------------------------------------------------------------------------
postpred_y <- postpred_samps$y.pred
post_y_summ <- t(apply(postpred_y, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
y_combn <- data.frame(y = dat_pred$y, yL = post_y_summ[, 1],
                      yM = post_y_summ[, 2], yU = post_y_summ[, 3])
library(ggplot2)
ggplot(data = y_combn, aes(x = y)) +
  geom_errorbar(aes(ymin = yL, ymax = yU), color = "skyblue") +
  geom_point(aes(y = yM), color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True y") + ylab("Posterior predictive of y") + theme_bw() +
  theme(panel.background = element_blank(), aspect.ratio = 1)

## -----------------------------------------------------------------------------
# training and test data sizes
n_train <- 100
n_pred <- 50

# load spatial Poisson data
data("simPoisson")
dat_train <- simPoisson[1:n_train, ]
dat_pred <- simPoisson[n_train + 1:n_pred, ]

## -----------------------------------------------------------------------------
mod1 <- spGLMstack(y ~ x1, data = dat_train, family = "poisson",
                   coords = as.matrix(dat_train[, c("s1", "s2")]), cor.fn = "matern",
                   params.list = list(phi = c(3, 4, 5), nu = c(0.5, 1.0),
                                      boundary = c(0.5)),
                   priors = list(nu.beta = 5, nu.z = 5),
                   n.samples = 1000,
                   loopd.controls = list(method = "CV", CV.K = 10, nMC = 500),
                   verbose = TRUE)

## -----------------------------------------------------------------------------
post_samps <- stackedSampler(mod1)

post_beta <- post_samps$beta
summary_beta <- t(apply(post_beta, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
rownames(summary_beta) <- mod1$X.names
print(summary_beta)

## -----------------------------------------------------------------------------
sp_pred <- as.matrix(dat_pred[, c("s1", "s2")])
X_new <- as.matrix(cbind(rep(1, n_pred), dat_pred$x1))

## -----------------------------------------------------------------------------
mod.pred <- posteriorPredict(mod1,
                             coords_new = sp_pred,
                             covars_new = X_new,
                             joint = FALSE)

## -----------------------------------------------------------------------------
postpred_samps <- stackedSampler(mod.pred)

## -----------------------------------------------------------------------------
postpred_z <- postpred_samps$z.pred
post_z_summ <- t(apply(postpred_z, 1, function(x) quantile(x, c(0.025, 0.5, 0.975))))
z_combn <- data.frame(z = dat_pred$z_true, zL = post_z_summ[, 1],
                      zM = post_z_summ[, 2], zU = post_z_summ[, 3])
ggplot(data = z_combn, aes(x = z)) +
  geom_errorbar(aes(ymin = zL, ymax = zU), color = "skyblue") +
  geom_point(aes(y = zM), color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True z") + ylab("Posterior predictive of z") + theme_bw() +
  theme(panel.background = element_blank(), aspect.ratio = 1)

