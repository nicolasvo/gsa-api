resource "aws_cloudwatch_log_group" "gsa" {
  name = "/aws/lambda/${aws_lambda_function.gsa.function_name}"

  retention_in_days = 7
}

resource "aws_lambda_function" "gsa" {
  function_name = "gsa"
  memory_size   = 8000
  timeout       = 900
  package_type  = "Image"
  architectures = ["x86_64"]
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_gsa}:${var.image_tag_gsa}"

  ephemeral_storage {
    size = 5000
  }

  environment {
    variables = {
      HOME = "/tmp"
    }
  }

  role = aws_iam_role.lambda.arn
}

resource "aws_lambda_function_url" "gsa" {
  function_name      = aws_lambda_function.gsa.function_name
  authorization_type = "NONE"
}

data "aws_iam_policy_document" "lambda" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "gsa"
  assume_role_policy = data.aws_iam_policy_document.lambda.json
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
  ]
}
