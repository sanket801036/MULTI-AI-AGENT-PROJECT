#  # Use the Jenkins image as the base image
# FROM jenkins/jenkins:lts
# # FROM jenkins/jenkins:2.462.3-lts-jdk17


# # Switch to root user to install dependencies
# USER root

# # Install prerequisites and Docker
# RUN apt-get update -y && \
#     apt-get install -y apt-transport-https ca-certificates curl gnupg software-properties-common && \
#     curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
#     echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
#     apt-get update -y && \
#     apt-get install -y docker-ce docker-ce-cli containerd.io && \
#     apt-get clean

# # Add Jenkins user to the Docker group (create if it doesn't exist)
# RUN groupadd -f docker && \
#     usermod -aG docker jenkins

# # Create the Docker directory and volume for DinD
# RUN mkdir -p /var/lib/docker
# VOLUME /var/lib/docker

# # Switch back to the Jenkins user
# USER jenkins


# Use the Jenkins image as the base image
FROM jenkins/jenkins:lts

# Switch to root user to install dependencies
USER root

# Install prerequisites and Docker CLI (Updated for Debian Trixie)
RUN apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg unzip && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker.gpg] https://download.docker.com/linux/debian trixie stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y docker-ce-cli && \
    apt-get clean

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

# Add Jenkins user to the Docker group
RUN groupadd -f docker && \
    usermod -aG docker jenkins

# Switch back to the Jenkins user
USER jenkins
