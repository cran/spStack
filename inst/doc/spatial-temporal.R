## -----------------------------------------------------------------------------
set.seed(1729)

## -----------------------------------------------------------------------------
library(spStack)
data("sim_stvcPoisson")
n_train <- 100
dat <- sim_stvcPoisson[1:n_train, ]

## -----------------------------------------------------------------------------
head(dat)

## -----------------------------------------------------------------------------
mod1 <- stvcGLMexact(y ~ x1 + (x1), data = dat, family = "poisson",
                     sp_coords = as.matrix(dat[, c("s1", "s2")]),
                     time_coords = as.matrix(dat[, "t_coords"]),
                     cor.fn = "gneiting-decay",
                     process.type = "independent",
                     priors = list(nu.beta = 5, nu.z = 5),
                     sptParams = list(phi_s = c(1, 2), phi_t = c(1, 2)),
                     verbose = FALSE, n.samples = 500)

## -----------------------------------------------------------------------------
mod1 <- recoverGLMscale(mod1)

## ----fig.align='center', fig.height=3.5, fig.width=7, fig.alt="Posterior distributions of the scale parameters."----
post_scale_df <- data.frame(value = sqrt(c(mod1$samples$z.scale[1, ], mod1$samples$z.scale[2, ])),
                            group = factor(rep(c("sigma.z1", "sigma.z2"), 
                                    each = length(mod1$samples$z.scale[1, ]))))
library(ggplot2)
ggplot(post_scale_df, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.6) +
  facet_wrap(~ group, scales = "free") + labs(x = "", y = "Density") +
  theme_bw() + theme(panel.background = element_blank(), 
                     panel.grid = element_blank(), aspect.ratio = 1)

## -----------------------------------------------------------------------------
mod2 <- stvcGLMexact(y ~ x1 + (x1), data = dat, family = "poisson",
                     sp_coords = as.matrix(dat[, c("s1", "s2")]),
                     time_coords = as.matrix(dat[, "t_coords"]),
                     cor.fn = "gneiting-decay",
                     process.type = "independent.shared",
                     priors = list(nu.beta = 5, nu.z = 5),
                     sptParams = list(phi_s = 1, phi_t = 1),
                     verbose = FALSE, n.samples = 500)

## -----------------------------------------------------------------------------
mod2 <- recoverGLMscale(mod2)

## ----fig.align='center', fig.height=3.5, fig.width=7, fig.alt="Posterior distributions of the scale parameters."----
post_scale_df <- data.frame(value = sqrt(mod2$samples$z.scale), 
                            group = factor(rep(c("sigma.z"), 
                                               each = length(mod2$samples$z.scale))))
ggplot(post_scale_df, aes(x = value)) +
  geom_density(fill = "lightblue", alpha = 0.6) +
  facet_wrap(~ group, scales = "free") + labs(x = "", y = "Density") +
  theme_bw() + theme(panel.background = element_blank(), 
                     panel.grid = element_blank(), aspect.ratio = 1)

## -----------------------------------------------------------------------------
mod3 <- stvcGLMexact(y ~ x1 + (x1), data = dat, family = "poisson",
                     sp_coords = as.matrix(dat[, c("s1", "s2")]),
                     time_coords = as.matrix(dat[, "t_coords"]),
                     cor.fn = "gneiting-decay",
                     process.type = "multivariate",
                     priors = list(nu.beta = 5, nu.z = 5),
                     sptParams = list(phi_s = 1, phi_t = 1),
                     verbose = FALSE, n.samples = 500)

## -----------------------------------------------------------------------------
mod3 <- recoverGLMscale(mod3)

## ----fig.align='center', fig.height=5, fig.width=4, fig.cap="Posterior distributions of elements of the scale matrix."----
post_scale_z <- mod3$samples$z.scale

r <- sqrt(dim(post_scale_z)[1])
# Function to get (i,j) index from row number (column-major)
get_indices <- function(k, r) {
  j <- ((k - 1) %/% r) + 1
  i <- ((k - 1) %% r) + 1
  c(i, j)
}

# Generate plots into a matrix
plot_matrix <- matrix(vector("list", r * r), nrow = r, ncol = r)
for (k in 1:(r^2)) {
  ij <- get_indices(k, r)
  i <- ij[1]
  j <- ij[2]
  
  if (i >= j) {
    df <- data.frame(value = post_scale_z[k, ])
    p <- ggplot(df, aes(x = value)) +
      geom_density(fill = "lightblue", alpha = 0.7) +
      theme_bw(base_size = 9) +
      labs(title = bquote(Sigma[.(i) * .(j)])) +
      theme(axis.title = element_blank(), axis.text = element_text(size = 6),
        plot.title = element_text(size = 9, hjust = 0.5),
        panel.grid = element_blank(), aspect.ratio = 1)
  } else {
    p <- ggplot() + theme_void()
  }
  
  plot_matrix[j, i] <- list(p)
}

library(patchwork)
# Assemble with patchwork
final_plot <- wrap_plots(plot_matrix, nrow = r)
final_plot

## -----------------------------------------------------------------------------
mod.list <- candidateModels(list(
  phi_s = list(1, 2, 3),
  phi_t = list(1, 2, 4),
  boundary = c(0.5, 0.75)), "cartesian")

## -----------------------------------------------------------------------------
mod1 <- stvcGLMstack(y ~ x1 + (x1), data = dat, family = "poisson",
                     sp_coords = as.matrix(dat[, c("s1", "s2")]),
                     time_coords = as.matrix(dat[, "t_coords"]),
                     cor.fn = "gneiting-decay",
                     process.type = "multivariate",
                     priors = list(nu.beta = 5, nu.z = 5),
                     candidate.models = mod.list,
                     loopd.controls = list(method = "CV", CV.K = 10, nMC = 500),
                     n.samples = 1000)

## -----------------------------------------------------------------------------
mod1 <- recoverGLMscale(mod1)

## -----------------------------------------------------------------------------
post_samps <- stackedSampler(mod1)

## ----fig.align='center', fig.height=3.5, fig.width=7--------------------------
post_z <- post_samps$z

post_z1_summ <- t(apply(post_z[1:n_train,], 1,
                        function(x) quantile(x, c(0.025, 0.5, 0.975))))
post_z2_summ <- t(apply(post_z[n_train + 1:n_train,], 1,
                        function(x) quantile(x, c(0.025, 0.5, 0.975))))

z1_combn <- data.frame(z = dat$z1_true, zL = post_z1_summ[, 1],
                       zM = post_z1_summ[, 2], zU = post_z1_summ[, 3])
z2_combn <- data.frame(z = dat$z2_true, zL = post_z2_summ[, 1],
                       zM = post_z2_summ[, 2], zU = post_z2_summ[, 3])

plot_z1_summ <- ggplot(data = z1_combn, aes(x = z)) +
  geom_errorbar(aes(ymin = zL, ymax = zU), alpha = 0.5, color = "skyblue") +
  geom_point(aes(y = zM), size = 0.5, color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True z1") + ylab("Posterior of z1") + theme_bw() +
  theme(panel.grid = element_blank(), aspect.ratio = 1)

plot_z2_summ <- ggplot(data = z2_combn, aes(x = z)) +
  geom_errorbar(aes(ymin = zL, ymax = zU), alpha = 0.5, color = "skyblue") +
  geom_point(aes(y = zM), size = 0.5, color = "darkblue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  xlab("True z2") + ylab("Posterior of z2") + theme_bw() +
  theme(panel.grid = element_blank(), aspect.ratio = 1)

ggpubr::ggarrange(plot_z1_summ, plot_z2_summ)

## ----fig.align='center', fig.height=5, fig.width=4, fig.cap="Stacked posterior distribution of the elements of the inter-process covariance matrix."----
post_scale_z <- post_samps$z.scale
r <- sqrt(dim(post_scale_z)[1])
# Generate plots into a matrix
plot_matrix <- matrix(vector("list", r * r), nrow = r, ncol = r)
for (k in 1:(r^2)) {
  ij <- get_indices(k, r)
  i <- ij[1]
  j <- ij[2]
  
  if (i >= j) {
    df <- data.frame(value = post_scale_z[k, ])
    p <- ggplot(df, aes(x = value)) +
      geom_density(fill = "lightblue", alpha = 0.7) +
      theme_bw(base_size = 9) +
      labs(title = bquote(Sigma[.(i) * .(j)])) +
      theme(axis.title = element_blank(), axis.text = element_text(size = 6),
        plot.title = element_text(size = 9, hjust = 0.5),
        panel.grid = element_blank(), aspect.ratio = 1)
  } else {
    p <- ggplot() + theme_void()
  }
  
  plot_matrix[j, i] <- list(p)
}

# Assemble with patchwork
final_plot <- wrap_plots(plot_matrix, nrow = r)
final_plot

