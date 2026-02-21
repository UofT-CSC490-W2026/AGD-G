resource "null_resource" "push_modal_secret" {
  triggers = {
    access_key = var.key_id
  }

  provisioner "local-exec" {
    command = <<EOT
      modal secret create --force aws \
        AWS_ACCESS_KEY_ID=${var.key_id} \
        AWS_SECRET_ACCESS_KEY=${var.key_secret}
    EOT
  }
}
